"""
Tool to train graph embeddings as detailed in "Retrofitting Manifolds to Semantic Graphs"
"""
import argparse
import json
import sys

from riemann.config.config_loader import initialize_config, get_config
import wandb
from riemann.graph_embedding_train_schedule import GraphEmbeddingTrainSchedule
from riemann.model import get_model
from riemann.data.data_loader import get_training_data


def train(args):
    # Initialize wandb dashboard
    wandb.init(project="retrofitting-manifolds")

    # Initialize Config
    initialize_config(args.config_file,
                      load_config=(args.config_file is not None),
                      config_updates=args.config_updates)

    # This command just preloads the training data.
    get_training_data()

    # Generate model
    model = get_model()

    # Train
    train_schedule = GraphEmbeddingTrainSchedule(model)
    train_schedule.train()


def transform_manifold(args):
    """
    Transform the manifold containing a given set of coordinates to one that
    better represents the distances on a given graph.
    """
    # Initialize Config
    initialize_config(args.config_file,
                      load_config=(args.config_file is not None),
                      config_updates=args.config_updates)
    print(json.dumps(get_config().as_json(), indent=2))


# noinspection DuplicatedCode
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--config_updates', type=str, default="",
                        help="Extra configuration to inject into config dict")
    parser.add_argument('-f', '--config_file', type=str, default=None,
                        help="File to load config from")

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('train', help=train.__doc__)
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('transform', help=transform_manifold.__doc__)
    command_parser.set_defaults(func=transform_manifold)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
