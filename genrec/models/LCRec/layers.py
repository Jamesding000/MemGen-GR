import numpy as np
from sklearn.cluster import KMeans
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        hidden_sizes (list): List of integers representing the sizes of hidden layers.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Attributes:
        mlp (nn.Sequential): Sequential container for the MLP layers.

    """

    def __init__(self, hidden_sizes: list, dropout: float = 0.0):
        super(MLP, self).__init__()
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            activation_func = nn.ReLU()
            if idx != len(hidden_sizes) - 2:
                mlp_modules.append(activation_func)
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.mlp(x)


class ResidualQuantizationLayer(nn.Module):
    """
    A module that performs residual quantization on input data.

    Args:
        n_codebooks (int): The number of codebooks to use.
        codebook_size (int or list): The size of the codebooks. If an int, the same codebook size is used for all levels.
            If a list, it should have length equal to n_codebooks, specifying the codebook size for each level.
        latent_size (int): The size of the latent space.
        low_usage_threshold (int): The threshold for low usage clusters.
        sk_epsilon (float, optional): The epsilon value for the Sinkhorn-Knopp algorithm. Default is 0.003.
        sk_iters (int, optional): The number of iterations for the Sinkhorn-Knopp algorithm. Default is 10.
        decay (float, optional): The decay factor for updating the codebooks. Default is 0.99.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
    """

    def __init__(self, n_codebooks, codebook_size, latent_size, low_usage_threshold, sk_epsilon=0.003, sk_iters=10, decay=0.99, eps=1e-5):
        super(ResidualQuantizationLayer, self).__init__()
        self.n_codebooks = n_codebooks
        self.latent_size = latent_size
        # Check if codebook_size is an int and convert it to a list of the same size for each level
        if isinstance(codebook_size, int):
            self.codebook_sizes = [codebook_size] * n_codebooks
        elif isinstance(codebook_size, list):
            if len(codebook_size) == n_codebooks:
                self.codebook_sizes = codebook_size
            else:
                raise ValueError("codebook_size must be an int or a list of int with length equal to n_codebooks")
        self.decay = decay
        self.eps = eps

        quantization_layers = []
        for codebook_size in self.codebook_sizes[:-1]:
            quantization_layers.append(QuantizationLayer(latent_size, codebook_size, low_usage_threshold, decay, eps))
        quantization_layers.append(QuantizationLayer(latent_size, self.codebook_sizes[-1], low_usage_threshold, decay, eps, sk_epsilon, sk_iters))
        self.quantization_layers = nn.ModuleList(quantization_layers)


    def forward(self, x: torch.Tensor, use_sk=False):
        """
        Forward pass of the residual quantization layer.

        Args:
            x (torch.Tensor): The input tensor.
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for quantization.

        Returns:
            output (torch.Tensor): The quantized output tensor.
            quantized_x (torch.Tensor): The sum of the quantized residuals.
            num_small_clusters (float): The number of small clusters encountered during quantization.
            sum_quant_loss (float): The sum of the quantization losses.
        """
        batch_size, _ = x.shape
        quantized_x = torch.zeros(batch_size, self.latent_size, device=x.device)
        sum_quant_loss = 0.0
        num_small_clusters = 0.0
        output = torch.empty(batch_size, self.n_codebooks, dtype=torch.long, device=x.device)
        for quantization_layer, level in zip(self.quantization_layers, range(self.n_codebooks)):
            if level == self.n_codebooks - 1:
                quant, quant_loss, n_small_clusters, output[:, level] = quantization_layer(x, use_sk)
            else:
                quant, quant_loss, n_small_clusters, output[:, level] = quantization_layer(x)
                
            x = x - quant
            quantized_x += quant
            sum_quant_loss += quant_loss
            num_small_clusters += n_small_clusters
        return output, quantized_x, num_small_clusters, sum_quant_loss

    def generate_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generates the codebook for each quantization layer.

        Args:
            x (torch.Tensor): The input tensor.
            device (torch.device): The device to use for the codebook.

        Returns:
            x (torch.Tensor): The updated input tensor after generating the codebook.
        """
        for quantization_layer in self.quantization_layers:
            x = quantization_layer.generate_codebook(x, device)
        return x


class QuantizationLayer(nn.Module):
    """
    A quantization layer that performs vector quantization on input data.

    Args:
        latent_size (int): The size of the input vectors.
        codebook_size (int): The number of codewords in the codebook.
        low_usage_threshold (int): The threshold for low usage clusters.
        decay (float, optional): The decay factor for updating the codebook. Defaults to 0.99.
        eps (float, optional): A small value added to avoid division by zero. Defaults to 1e-5.
        sk_epsilon (float, optional): The epsilon value for the Sinkhorn-Knopp algorithm. Defaults to -1.0.
        sk_iters (int, optional): The number of iterations for the Sinkhorn-Knopp algorithm. Defaults to 0.
    """

    def __init__(self, latent_size, codebook_size, low_usage_threshold, decay=0.99, eps=1e-5, sk_epsilon=-1.0, sk_iters=0):
        super(QuantizationLayer, self).__init__()
        self.dim = latent_size
        self.n_embed = codebook_size
        self.decay = decay
        self.eps = eps
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        embed = torch.zeros(latent_size, codebook_size)
        self.embed = torch.nn.Parameter(embed, requires_grad=False)
        self.low_usage_threshold = low_usage_threshold
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    @staticmethod
    def center_distance(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    @torch.no_grad()
    def sinkhorn(self, distances, epsilon=0.003, iterations=1):
        Q = torch.exp(- distances / epsilon)

        B = Q.shape[0]  # number of samples to assign
        K = Q.shape[1]  # how many centroids per block (usually set to 256)

        # make the matrix sums to 1
        sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)

        Q /= sum_Q
        for _ in range(iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_0 = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_0
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q

    def forward(self, x: torch.Tensor, use_sk=False):
        """
        Forward pass of the quantization layer.

        Args:
            x (torch.Tensor): The input tensor.
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for quantization.

        Returns:
            tuple: A tuple containing the quantized tensor, quantization loss, number of small clusters, and embedding indices.
        """
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        
        if (self.training and self.use_sk) or use_sk:
            dist = self.center_distance(dist)
            dist = dist.double()
            Q = self.sinkhorn(dist, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise RuntimeError(f"Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(Q, dim=-1)
        else:
            _, embed_ind = (-dist).max(1)
        

        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(x.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = x.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            # reassign low usage entries
            small_clusters = self.cluster_size < 1.0
            n_small_clusters = small_clusters.sum().item()
            if self.low_usage_threshold != -1 and n_small_clusters > self.low_usage_threshold:
                sampled_indices = torch.randint(0, x.size(0), (n_small_clusters,), device=x.device)
                sampled_values = x[sampled_indices].clone().detach()
                self.embed[:, small_clusters] = sampled_values.T
                self.embed_avg[:, small_clusters] = self.embed[:, small_clusters].clone()
                self.cluster_size[small_clusters] = 0.99
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        else:
            small_clusters = self.cluster_size < 1.0
            n_small_clusters = small_clusters.sum().item()

        quant_loss = torch.nn.functional.mse_loss(quantize.detach(), x)
        quantize = (x + (quantize - x).detach())
        return quantize, quant_loss, n_small_clusters, embed_ind

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """
        Embeds the given indices using the codebook.

        Args:
            embed_id (torch.Tensor): The embedding indices.

        Returns:
            torch.Tensor: The embedded vectors.
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def encode_to_id(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input vectors to embedding indices.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The embedding indices.
        """
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])

        return embed_ind

    def generate_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generates the codebook using K-means clustering.

        Args:
            x (torch.Tensor): The input tensor.
            device (torch.device): The device to use for computations.

        Returns:
            torch.Tensor: The residual tensor after quantization.
        """
        kmeans = KMeans(n_clusters=self.n_embed, n_init='auto').fit(x.detach().cpu().numpy())
        self.embed.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).view(self.dim, self.n_embed)
        self.embed_avg.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).view(self.dim, self.n_embed)
        self.cluster_size.data = torch.tensor(np.bincount(kmeans.labels_), dtype=torch.float, device=device)
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)
        return x - quantize


class RQVAEModel(nn.Module):
    """
    RQVAE (Residual Quantized Variational Autoencoder) class.

    Args:
        hidden_sizes (list): List of integers specifying the sizes of hidden layers in the encoder and decoder.
        n_codebooks (int): Number of codebooks used for quantization.
        codebook_size (int): Size of each codebook.
        dropout (float): Dropout probability applied to the hidden layers.
        low_usage_threshold (int): Threshold for low usage clusters.

    Attributes:
        encoder (MLP): Multi-layer perceptron used for encoding the input.
        quantization_layer (ResidualQuantizationLayer): Residual quantization layer.
        decoder (MLP): Multi-layer perceptron used for decoding the quantized input.

    Methods:
        forward(x): Performs forward pass through the RQVAE model.
        encode(x): Encodes the input into a latent representation.
        generate_codebook(x, device): Generates the codebook for quantization.
    """

    def __init__(
        self,
        hidden_sizes: list,
        n_codebooks: int,
        codebook_size: Union[int, list],
        dropout: float,
        low_usage_threshold: int,
        sk_epsilon: float,
        sk_iters: int
    ):
        super(RQVAEModel, self).__init__()
        self.encoder = MLP(hidden_sizes, dropout=dropout)
        self.quantization_layer = ResidualQuantizationLayer(n_codebooks, codebook_size, hidden_sizes[-1], low_usage_threshold, sk_epsilon, sk_iters)
        self.decoder = MLP(hidden_sizes[::-1], dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Performs a forward pass through the RQVAE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            decoded (torch.Tensor): Decoded output tensor.
            quant_loss (torch.Tensor): Quantization loss.
            num_small_clusters (int): Number of small clusters in the quantization layer.

        """
        encoded = self.encoder(x)
        _, quantized_x, num_small_clusters, quant_loss = self.quantization_layer(encoded)
        decoded = self.decoder(quantized_x)
        return decoded, quant_loss, num_small_clusters

    def encode(self, x: torch.Tensor, use_sk=False) -> np.ndarray:
        """
        Encodes the input into a latent representation.

        Args:
            x (torch.Tensor): Input tensor.
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for quant

        Returns:
            output (numpy.ndarray): Encoded output as a numpy array.

        """
        encoded = self.encoder(x)
        output, _, _, _ = self.quantization_layer(encoded, use_sk)
        return output.detach().cpu().numpy()

    def generate_codebook(self, x: torch.Tensor, device: torch.device):
        """
        Generates the codebook for quantization.

        Args:
            x (torch.Tensor): Input tensor.
            device (torch.device): Device to be used for codebook generation.

        """
        encoded = self.encoder(x)
        self.quantization_layer.generate_codebook(encoded, device)
