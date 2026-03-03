import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from genrec.utils import parse_command_line_args, get_pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TIGER')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--experiment_tuple', type=str, required=True)
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    print('args', args)
    print('unparsed_args', unparsed_args)
    print('command_line_configs', command_line_configs)

    # process experiment tuple
    parts = args.experiment_tuple.split(':')
    assert len(parts) == 3, f"Invalid experiment_tuple format: {args.experiment_tuple}"
    cb_size, n_cb, budget = int(parts[0]), int(parts[1]), int(parts[2])

    base_results_dir = "logs/fine_grained_results"
    project_name = os.environ["WANDB_PROJECT"]
    project_results_dir = os.path.join(base_results_dir, project_name)
    os.makedirs(project_results_dir, exist_ok=True)
    eval_results_file = os.path.join(project_results_dir, f"codebook_{cb_size}x{n_cb}.csv")

    command_line_configs['rq_codebook_size'] = cb_size
    command_line_configs['rq_n_codebooks'] = n_cb
    command_line_configs['budget_epochs'] = budget
    command_line_configs['wandb_project'] = project_name
    command_line_configs['wandb_run_name'] = f"{cb_size}x{n_cb}"
    command_line_configs['eval_results_file'] = eval_results_file

    # run pipeline
    pipeline = get_pipeline(args.model)(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config,
        config_dict=command_line_configs
    )
    
    pipeline.run()