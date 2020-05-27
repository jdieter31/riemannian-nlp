"""
Tool to train graph embeddings as detailed in "Retrofitting Manifolds to Semantic Graphs"
"""
import argparse
import json
import sys

from riemann.config.config import ConfigDictParser
from riemann.config.config_loader import initialize_config, get_config
import wandb

from riemann.graph_embedder import GraphEmbedder
from riemann.graph_embedding_train_schedule import GraphEmbeddingTrainSchedule
from riemann.model import get_model, torch
from riemann.data.data_loader import get_training_data
from riemann.visualize import plot


def train(args):
    # Initialize wandb dashboard
    wandb.init(project="retrofitting-manifolds")

    # Initialize Config
    initialize_config(args.config_file,
                      load_config=(args.config_file is not None),
                      config_updates=ConfigDictParser.parse(args.config_updates))
    # This command just preloads the training data.
    get_training_data()

    # Generate model
    model = get_model()

    # Train
    train_schedule = GraphEmbeddingTrainSchedule(model)
    train_schedule.train()

    # Save the model
    if args.model_file:
        model.to_file(args.model_file)


def plot_transformation(args):
    """
    Plots the manifold transformation learned by the given model.
    better represents the distances on a given graph.
    """
    model = GraphEmbedder.from_file(args.model_file)
    # Run plot with the data
    fig = plot(model)
    fig.show()


# noinspection DuplicatedCode
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('train', help=train.__doc__)
    command_parser.add_argument('-u', '--config_updates', type=str, default="",
                                help="Extra configuration to inject into config dict")
    command_parser.add_argument('-f', '--config_file', type=str, default=None,
                                help="File to load config from")
    command_parser.add_argument('-m', '--model_file', type=str, default=None,
                                help="Path to save model at")

    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('plot', help=plot_transformation.__doc__)
    command_parser.add_argument('model_file', type=str,
                                help="File to load model from")
    command_parser.set_defaults(func=plot_transformation)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
