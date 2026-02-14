import argparse

from genrec.utils import parse_command_line_args, get_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TIGER', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    pipeline = get_pipeline(args.model)(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_file=args.config,
        config_dict=command_line_configs
    )
    pipeline.run()
