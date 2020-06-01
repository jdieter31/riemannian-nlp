from typing import List, Callable, cast, Sized

import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm

from .config.config_loader import get_config
from .config.config_specs.model_config import ModelConfig
from .data.batching import DataBatch
from .data.data_loader import get_training_data, get_eval_data
from .evaluations.mean_rank import run_evaluation as run_mean_rank_evaluation
from .featurizers.graph_object_id_featurizer_embedder import GraphObjectIDFeaturizerEmbedder
from .graph_embedder import GraphEmbedder
from .losses.grad_norm_loss_processor import GradNormLossProcessor
from .losses.graph_manifold_margin_loss import graph_manifold_margin_loss
from .losses.single_loss_processor import SingleLossProcessor
from .manifold_tensors import ManifoldParameter
from .optimizer_gen import get_optimizer
from .train import TrainSchedule
from .visualize import plot, plot_degree_distribution


class GraphEmbeddingTrainSchedule(TrainSchedule):
    """
    Training schedule for graph embedding
    """

    def __init__(self, model: GraphEmbedder):
        super(GraphEmbeddingTrainSchedule, self).__init__()

        self.model = model
        self.training_data = get_training_data()
        self.eval_data = get_eval_data()
        self.optimizer = get_optimizer()
        self._loss_processor = None
        self.tasks = [self._get_loss_processor()]
        self._add_cyclic_evaluations()
        self._add_neighbor_resamplings()

    def epoch_iterator(self):
        general_config = get_config().general
        sampling_config = get_config().sampling
        for i in tqdm(range(general_config.n_epochs), desc="Epochs",
                      dynamic_ncols=True):
            # Epoch loop

            # Return iterator of data and training tasks to complete for each
            # DataBatch
            data_iterator = self.training_data.get_neighbor_iterator(
                sampling_config.train_sampling_config)
            n_tasks = len(cast(Sized, data_iterator))
            yield data_iterator, (self._get_tasks_for_batch() for _ in range(n_tasks))

        model_config = get_config().model
        if model_config.save_dir is not None:
            self.model.to_file(f"{model_config.save_dir}end_training.zip")


    def _get_loss_processor(self):
        if self._loss_processor is None:
            losses = self._get_losses()
            if len(losses) == 1:
                self._loss_processor = SingleLossProcessor(losses[0], self.optimizer)
            else:
                assert isinstance(self.model, GraphObjectIDFeaturizerEmbedder)
                params = [param for param in self.model.model.parameters()
                          if not isinstance(param, ManifoldParameter)]
                self._loss_processor = GradNormLossProcessor(losses, self.optimizer, params)

        return self._loss_processor

    def _get_losses(self) -> List[Callable[[DataBatch], torch.Tensor]]:
        model_config: ModelConfig = get_config().model
        loss_config = get_config().loss
        manifold = model_config.target_manifold_.get_manifold_instance()

        margin_loss = lambda data_batch: \
            graph_manifold_margin_loss(self.model, data_batch, manifold,
                                       loss_config.margin)
        return [margin_loss] + self.model.get_losses()

    def _add_cyclic_evaluations(self):
        eval_config = get_config().eval

        if eval_config.eval_link_pred:
            def task():
                print(f"running lnk_pred evaluation {self.iteration_num}")
                eval_data = get_eval_data()
                eval_data.add_manifold_nns(self.model)
                run_mean_rank_evaluation(self, "lnk_pred",
                                         step=self.iteration_num)
            self.add_cyclic_task(task, eval_config.link_pred_frequency)

        if eval_config.eval_reconstruction:
            def task():
                print(f"running reconst evaluation {self.iteration_num}")
                train_data = get_training_data()
                train_data.add_manifold_nns(self.model)
                run_mean_rank_evaluation(self, "reconstr",
                                            step=self.iteration_num,
                                            reconstruction=True)
            self.add_cyclic_task(task, eval_config.reconstruction_frequency)

        if eval_config.make_visualization and isinstance(self.model,
                                                         GraphObjectIDFeaturizerEmbedder):
            def visualize_task():
                fig = plot(cast(GraphObjectIDFeaturizerEmbedder, self.model))
                wandb.log({"visualization": wandb.Image(fig)})
                plt.close(fig)

            self.add_cyclic_task(visualize_task,
                                 eval_config.visualization_frequency)

    def _add_neighbor_resamplings(self):
        sampling_config = get_config().sampling
        if sampling_config.train_sampling_config.n_manifold_neighbors > 0 or \
                sampling_config.eval_sampling_config.n_manifold_neighbors > 0:

            def task():
                train_data = get_training_data()
                train_data.add_manifold_nns(self.model)

            self.add_cyclic_task(task,
                                 sampling_config.manifold_neighbor_resampling_rate,
                                 cycle_on_iterations=False)

    def _get_tasks_for_batch(self):
        for task in self.tasks:
            yield task
