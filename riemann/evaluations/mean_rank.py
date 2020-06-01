import torch
import wandb

from ..config.config_loader import get_config
from ..config.config_specs.model_config import ModelConfig
from ..data.batching import BatchTask
from ..data.data_loader import get_training_data, get_eval_data
from ..data.graph_data_batch import GraphDataBatch
from ..model import get_model
from ..train import DummyTrainSchedule

step_num = None
bests = {}


def compute_map(ranks):
    """
    Returns the Mean Averaged Precision given a list of ranks for positive instances.
    :param ranks: an array with the rank indices of positive elements.
    :return:
    """
    # The precision at k is true positives / total predictions;
    # the numerator is the index in `ranks`, the denominator is the value of rank
    precision_at_k = torch.arange(1, len(ranks)+1, dtype=torch.float32)/ranks
    return precision_at_k.mean()


def compute_mrr(ranks, adjust_ranks: bool = True):
    # The rank at an index ignores positive elements
    # (hence subtracting the number of positive elements) here.
    if adjust_ranks:
        ranks = ranks - torch.arange(len(ranks), dtype=torch.float32)
    rr = 1. / ranks
    return rr.mean()


class MeanRankEvaluator(BatchTask):

    def __init__(self):
        """
        Initializes mean rank averaging
        """

        self.ranks_computed = 0
        self.mean_rank_sum = 0
        self.mrr_sum = 0
        self.map_sum = 0
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
        #: Anything with train_distance >=2 is a negative example
        n_neighbors = (train_distances < 2).sum(dim=-1)

        for row in (sorted_indices < n_neighbors.unsqueeze(1)):
            ranks = (row.nonzero().squeeze(-1) + 1).cpu().to(torch.float32)
            if len(ranks) > 0:
                adjusted_ranks = (ranks - torch.arange(len(ranks), dtype=torch.float32))
                self.hitsat10 += (ranks <= 10).sum().numpy()
                self.mean_rank_sum += adjusted_ranks.mean().numpy()
                self.mrr_sum += compute_mrr(adjusted_ranks, adjust_ranks=False)
                self.map_sum += compute_map(ranks)
            self.ranks_computed += 1

    def finish_computations_and_log(self, log_name, log_results=True):
        hitsat10 = self.hitsat10 / self.ranks_computed
        mean_rank = self.mean_rank_sum / self.ranks_computed
        mean_rec_rank = self.mrr_sum / self.ranks_computed
        mean_map = self.map_sum / self.ranks_computed

        if log_results:
            wandb.log({f"{log_name}/hitsat10": hitsat10}, step=step_num)
            wandb.log({f"{log_name}/mean_rank": mean_rank}, step=step_num)
            wandb.log({f"{log_name}/mean_rec_rank": mean_rec_rank}, step=step_num)
            wandb.log({f"{log_name}/map": mean_map}, step=step_num)

        return mean_rank, mean_rec_rank, hitsat10, mean_map


def run_evaluation(train_schedule=None,
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

    if train_schedule is None:
        print_evaluation = True
        train_schedule = DummyTrainSchedule()
    else:
        print_evaluation = False

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
    mean_rank, mean_rec_rank, hitsat10, mean_map = \
            mean_rank_evaluator.finish_computations_and_log(
                log_name, log_results=not print_evaluation)

    if print_evaluation:
        print(f"{log_name}/mean_rank: {mean_rank}")
        print(f"{log_name}/mean_rec_rank: {mean_rec_rank}")
        print(f"{log_name}/hitsat10: {hitsat10}")
        print(f"{log_name}/mean_map: {mean_map}")
    else:
        if f"{log_name}/mean_rank" not in bests or bests[f"{log_name}/mean_rank"] > mean_rank:
            bests[f"{log_name}/mean_rank"] = mean_rank
            model_config = get_config().model
            wandb.run.summary[f"best_{log_name}/mean_rank"] = mean_rank
            if  model_config.save_dir is not None:
                train_schedule.model.to_file(f"{model_config.save_dir}best_{log_name}_mean_rank.zip")

        if f"{log_name}/hitsat10" not in bests or bests[f"{log_name}/hitsat10"] < hitsat10:
            bests[f"{log_name}/hitsat10"] = hitsat10
            model_config = get_config().model
            wandb.run.summary[f"best_{log_name}/hitsat10"] = hitsat10
            if  model_config.save_dir is not None:
                train_schedule.model.to_file(f"{model_config.save_dir}best_{log_name}_hitsat10.zip")

        if f"{log_name}/mean_rec_rank" not in bests or bests[f"{log_name}/mean_rec_rank"] < mean_rec_rank:
            bests[f"{log_name}/mean_rec_rank"] = mean_rec_rank
            model_config = get_config().model
            wandb.run.summary[f"best_{log_name}/mean_rec_rank"] = mean_rec_rank
            if  model_config.save_dir is not None:
                train_schedule.model.to_file(f"{model_config.save_dir}best_{log_name}_mean_rec_rank.zip")

        if f"{log_name}/mean_map" not in bests or bests[f"{log_name}/mean_map"] < mean_map:
            bests[f"{log_name}/mean_map"] = mean_map
            model_config = get_config().model
            wandb.run.summary[f"best_{log_name}/mean_map"] = mean_map
            if  model_config.save_dir is not None:
                train_schedule.model.to_file(f"{model_config.save_dir}best_{log_name}_mean_map.zip")
