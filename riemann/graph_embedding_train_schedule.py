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

from tqdm import tqdm
import torch

class GraphEmbeddingTrainSchedule(TrainSchedule):
    """
    Training schedule for graph embedding
    """

    def __init__(self, model: GraphEmbedder):
        self.model = model
        self.training_data = get_training_data()
        self.eval_data = get_eval_data()
        self.epoch_num = 0
        self.iteration_num = 0
        self.optimizer = get_optimizer()
        self.tasks = [self._get_loss_processor()]

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

            self.epoch_num += 1

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
                                           margin) 
        return [margin_loss]

    def _get_tasks_for_batch(self):
        for task in self.tasks:
            yield task

        self.iteration_num += 1
