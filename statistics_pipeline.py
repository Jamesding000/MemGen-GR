import argparse

from genrec.pipeline import Pipeline
from genrec.fine_grained_evaluator import FineGrainedEvaluator
from genrec.utils import parse_command_line_args


class StatisticPipeline(Pipeline):
    def __init__(
        self,
        model_name,
        dataset_name,
        checkpoint_path: str = None,
        tokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            trainer=trainer,
            config_dict=config_dict,
            config_file=config_file
        )
        
        self.max_hop = 5
        self.evaluator = FineGrainedEvaluator(
            train_item_seqs=self.split_datasets['train']['item_seq'],
            max_hop=self.max_hop
        )

    def run(self):
        # Compute statistics using FineGrainedEvaluator
        ratios = self.evaluator.compute_pattern_statistics(self.split_datasets['test']['item_seq'])
        
        # Print results grouped by hop
        for hop in range(2, self.max_hop + 1):
            self.log(f'=== Hop {hop} ===')
            for logic in list(self.evaluator.logic2judger.keys()) + ['novelty']:
                label = f'{logic}_{hop}'
                percentage = ratios[label]
                self.log(f'{logic}: {percentage:.6f}')
            self.log('')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    pipeline = StatisticPipeline(
        model_name='SASRec',
        dataset_name=args.dataset,
        config_dict=command_line_configs
    )
    pipeline.run()
