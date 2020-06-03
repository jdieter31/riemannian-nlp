"""
Tool to train graph embeddings as detailed in "Retrofitting Manifolds to Semantic Graphs"
"""
import argparse
import json
import sys
from typing import cast

from riemann.config.config import ConfigDictParser
from riemann.config.config_loader import initialize_config, get_config
import wandb

from riemann.featurizers.graph_object_id_featurizer_embedder import GraphObjectIDFeaturizerEmbedder
from riemann.graph_embedder import GraphEmbedder
from riemann.graph_embedding_train_schedule import GraphEmbeddingTrainSchedule
from riemann.model import get_model
from riemann.data.data_loader import get_training_data, get_eval_data
from riemann.visualize import plot, plot_input, plot_output
from riemann.evaluations.mean_rank import run_evaluation as run_mean_rank_evaluation
from riemann.config.config_loader import get_config


def train(args):
    # Initialize Config
    initialize_config(args.config_file,
                      load_config=(args.config_file is not None),
                      config_updates=ConfigDictParser.parse(args.config_updates))
    # Log this configuration to wandb
    # Initialize wandb dashboard
    config = get_config()
    if config.loss.use_proximity_regularizer:
        loss_description = "P"
    elif config.loss.use_conformality_regularizer:
        loss_description = f"C{config.loss.conformality:0.2f}"
    else:
        loss_description = "N"

    wandb.init(project="retrofitting-manifolds",
               name=f"{config.model.intermediate_manifold}^{config.model.intermediate_layers}"
                    f"->{config.model.target_manifold}{loss_description}",
               config=get_config().as_json(),
               group="ToyVisualizations")

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


def eval_model(args):
    # Initialize Config
    initialize_config(args.config_file,
                      load_config=(args.config_file is not None),
                      config_updates=ConfigDictParser.parse(args.config_updates))

    eval_config = get_config().eval
    model = get_model()

    sampling_config = get_config().sampling
    if sampling_config.train_sampling_config.n_manifold_neighbors > 0 or \
            sampling_config.eval_sampling_config.n_manifold_neighbors > 0:

        train_data = get_training_data()
        train_data.add_manifold_nns(model)

        eval_data = get_eval_data()
        if eval_data is not None:
            # Hacky way of not having to generate this again
            eval_data.manifold_nns = train_data.manifold_nns

    if eval_config.eval_link_pred:
        run_mean_rank_evaluation(None, "lnk_pred")
    if eval_config.eval_reconstruction:
        run_mean_rank_evaluation(None, "reconstr", reconstruction=True)


def plot_transformation(args):
    """
    Plots the manifold transformation learned by the given model.
    better represents the distances on a given graph.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    MEDIUM_SIZE = 18
    BIGGER_SIZE = 24

    plt.rc('text', usetex=True)              # controls default text sizes
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('font', family="serif")         # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

    model: GraphObjectIDFeaturizerEmbedder = cast(GraphObjectIDFeaturizerEmbedder,
                                                      GraphEmbedder.from_file(args.model_file))

    inputs_tensor = model.get_featurizer_graph_embedder().retrieve_nodes(
        model.graph_dataset.n_nodes()
    )
    output_tensor = model.retrieve_nodes(model.graph_dataset.n_nodes())

    inputs = inputs_tensor.detach().numpy()
    outputs = output_tensor.detach().numpy()

    #mpl.style.use('seaborn')

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # plot_input(ax, model, inputs)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # fig.tight_layout()
    # fig.show()
    # input("Press any key to exit.")

    fig = plt.figure(figsize=(8, 8))

    if outputs.shape[-1] == 2:
        ax = fig.add_subplot(111)
        plot_output(ax, model, inputs, outputs)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
    else:
        assert outputs.shape[-1] == 3
        ax = fig.add_subplot(111, projection='3d')
        plot_output(ax, model, inputs, outputs)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    fig.tight_layout()
    fig.show()

    input("Press any key to exit.")


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

    command_parser = subparsers.add_parser('eval')
    command_parser.add_argument('-u', '--config_updates', type=str, default="",
                                help="Extra configuration to inject into config dict")
    command_parser.add_argument('-f', '--config_file', type=str, default=None,
                                help="File to load config from")
    command_parser.add_argument('-m', '--model_file', type=str, default=None,
                                help="Path to save model at")

    command_parser.set_defaults(func=eval_model)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
