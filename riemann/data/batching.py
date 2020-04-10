import abc
from typing import Dict
import torch

class DataBatch(abc.ABC):
    """
    Abstract class for batching torch data
    """

    @abc.abstractmethod
    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of tensors for data in this batch.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_data_type(cls) -> str:
        """
        Returns what type of data is contained in this batch i.e. "graph" data
        or "classification"
        """
        raise NotImplementedError


class BatchTask(abc.ABC):
    """
    Abstract class for a task that does something when fed a batch of data
    (i.e. runs an optimizer, computes an evaluation, etc)
    """

    @abc.abstractmethod
    def process_batch(data_batch: DataBatch):
        raise NotImplementedError

class FunctionBatchTask(BatchTask):
    """
    Basic batch task that just executes a function
    """
    def __init__(self, function):
        self.function = function

    def process_batch(data_batch: DataBatch):
        self.function(data_batch)

    


