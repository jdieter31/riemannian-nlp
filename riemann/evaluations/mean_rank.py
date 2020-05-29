import torch
import wandb

from ..config.config_loader import get_config
from ..config.config_specs.model_config import ModelConfig
from ..data.batching import BatchTask
from ..data.data_loader import get_training_data, get_eval_data
from ..data.graph_data_batch import GraphDataBatch
from ..model import get_model

step_num = None
bests = {}


class MeanRankEvaluator(BatchTask):

    def __init__(self):
        """
        Initializes mean rank averaging
        """

        self.ranks_computed = 0
        self.rank_sum = 0
        self.rec_rank_sum = 0
        self.hitsat10 = 0

    def process_batch(self, batch: GraphDataBatch):
        """
        Processes a batch as documented in superclass
        """

        # Get the model
        model = get_model()

        # Get manifold
        model_config: ModelConfig = get_config().model
        manifold = model_config.target_manifold_.get_manifold_instance()

        # Isolate portion of input that are neighbors
        sample_vertices = model.embed_nodes(batch.get_tensors()["neighbors"])

        # Isolate portion of input that are main vertices
        main_vertices = \
            model.embed_nodes(batch.get_tensors()["vertices"]
                              ).unsqueeze(1).expand_as(sample_vertices)

        manifold_dists = manifold.dist(main_vertices, sample_vertices)

        train_distances = \
            batch.get_tensors()["train_distances"].to(main_vertices.device)

        sorted_indices = manifold_dists.argsort(dim=-1)
        manifold_dists_sorted = torch.gather(manifold_dists, -1, sorted_indices)
        n_neighbors = (train_distances < 2).sum(dim=-1)
        batch_nums, neighbor_ranks = (sorted_indices <
                                      n_neighbors.unsqueeze(1)).nonzero(as_tuple=True)
        neighbor_ranks += 1

        adjust_indices = torch.arange(n_neighbors.max())
        neighbor_adjustements = torch.cat([adjust_indices[:n_neighbors[i]] \
                                           for i in range(n_neighbors.size(0))])
        neighbor_ranks -= neighbor_adjustements.to(neighbor_ranks.device)
        neighbor_ranks = neighbor_ranks.float()
        rec_ranks = 1 / neighbor_ranks
        self.hitsat10 += (neighbor_ranks <= 10).sum().cpu().numpy()
        self.rank_sum += neighbor_ranks.sum().cpu().numpy()
        self.rec_rank_sum += rec_ranks.sum().cpu().numpy()
        self.ranks_computed += neighbor_ranks.size(0)

    def finish_computations_and_log(self, log_name):
        mean_rank = self.rank_sum / self.ranks_computed
        mean_rec_rank = self.rec_rank_sum / self.ranks_computed
        hitsat10 = self.hitsat10 / self.ranks_computed

        wandb.log({f"{log_name}/mean_rank": mean_rank}, step=step_num)
        wandb.log({f"{log_name}/mean_rec_rank": mean_rec_rank}, step=step_num)
        wandb.log({f"{log_name}/hitsat10": hitsat10}, step=step_num)

        return mean_rank, mean_rec_rank, hitsat10


def run_evaluation(train_schedule,
                   log_name="",
                   reconstruction=False,
                   step=None):
    """
    Runs this evaluation using a train schedule

    Params:
        train_schedule (TrainSchedule): TrainSchedule to run the epoch through
        log_name (str): Prefix to add to log names for wandb
        reconstruction (bool): Evaluate reconstruction of training data even if
            eval data is available. In this setting edges in the eval dataset
            will be counted as negative since only reconstruction of the
            training data is to be measured.
        step (int): Step number for logging to wandb
    """
    global step_num
    global bests
    step_num = step

    sampling_config = get_config().sampling
    eval_config = get_config().eval

    eval_data = get_eval_data()

    # If eval_data is None we should still evaluate mean ranks on the training
    # data (or if in the reconstruction setting)
    if eval_data is None or reconstruction:
        eval_data = get_training_data()

    data_iterator = \
        eval_data.get_neighbor_iterator(sampling_config.eval_sampling_config,
                                        data_fraction=eval_config.data_fraction)
    mean_rank_evaluator = MeanRankEvaluator()
    epoch = (data_iterator, ([mean_rank_evaluator] for _ in
                             range(len(data_iterator))))
    train_schedule.run_epoch(epoch,
                             prog_desc=f"{log_name}",
                             count_iterations=False)
    mean_rank, mean_rec_rank, hitsat10 = mean_rank_evaluator.finish_computations_and_log(log_name)

    if log_name not in bests or bests[log_name] > mean_rank:
        bests[log_name] = mean_rank
        model_config = get_config().model
        wandb.run.summary[f"best_{log_name}"] = mean_rank
        if  model_config.save_dir is not None:
            train_schedule.model.to_file(f"{model_config.save_dir}best_{log_name}.zip")

