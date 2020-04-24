from .train import TrainSchedule 
from .data.batching import BatchTask, DataBatch
from .data.graph_dataset import GraphDataset
from .graph_embedder import GraphEmbedder
from .data.data_loader import get_training_data, get_eval_data
from .config.config_loader import get_config
from .data.batching import FunctionBatchTask
from typing import List, Callable
from .losses.grad_norm_loss_processor import GradNormLossProcessor
from .losses.single_loss_processor import SingleLossProcessor
from .losses.graph_manifold_margin_loss import graph_manifold_margin_loss
from .optimizer_gen import get_optimizer
from .evaluations.mean_rank import run_evaluation as run_mean_rank_evaluation
import wandb

from tqdm import tqdm
import torch

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
            yield data_iterator, (self._get_tasks_for_batch() for _ in
                                  range(len(data_iterator)))

    def _get_loss_processor(self):
        losses = self._get_losses()
        if len(losses) == 1:
            return SingleLossProcessor(losses[0], self.optimizer)
        else:
            return GradNormLossProcessor(losses, self.optimizer)

    def _get_losses(self) -> List[Callable[[DataBatch], torch.Tensor]]:
        general_config = get_config().general
        loss_config = get_config().loss
        manifold = general_config.embed_manifold.get_manifold_instance()

        margin_loss = lambda data_batch: \
                graph_manifold_margin_loss(self.model, data_batch, manifold,
                                           loss_config.margin) 
        return [margin_loss]

    def _add_cyclic_evaluations(self):
        eval_config = get_config().eval
        
        if eval_config.eval_link_pred:
            task = lambda : run_mean_rank_evaluation(self, "lnk_pred",
                                                     step=self.iteration_num)
            self.add_cyclic_task(task, eval_config.link_pred_frequency)

        if eval_config.eval_reconstruction:
            task = lambda : run_mean_rank_evaluation(self, "reconstr",
                                                     step=self.iteration_num,
                                                     reconstruction=True)
            self.add_cyclic_task(task, eval_config.reconstruction_frequency)

    def _add_neighbor_resamplings(self):
        sampling_config = get_config().sampling
        if sampling_config.train_sampling_config.n_manifold_neighbors > 0 or \
            sampling_config.eval_sampling_config.n_manifold_neighbors > 0:

            def task():
                train_data = get_training_data()
                train_data.add_manifold_nns(self.model)

                eval_data = get_eval_data()
                if eval_data is not None:
                    # Hacky way of not having to generate this again
                    eval_data.manifold_nns = train_data.manifold_nns
            
            self.add_cyclic_task(task,
                                 sampling_config.manifold_neighbor_resampling_rate,
                                 cycle_on_iterations=False)



    def _get_tasks_for_batch(self):
        for task in self.tasks:
            yield task
