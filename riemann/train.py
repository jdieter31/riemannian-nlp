import abc 
from typing import Iterator, Tuple, List
from .data.batching import DataBatch, BatchTask
from tqdm import tqdm

class TrainSchedule(abc.ABC):
    """
    Abstract class that runs a training operation choosing which dataset to use
    each epoch as well as what tasks to run for each batch of the data. (i.e.
    train one loss, train another, potentially run an evaluation)
    """

    @abc.abstractmethod
    def epoch_iterator(self) -> Iterator[Tuple[Iterator[DataBatch],
                                               Iterator[List[BatchTask]]]]:
        """
        Should return an iterator that sequentially yields each new epoch of
        data. Each epoch consists of an Iterator of batches in the format of a
        DataBatch as well as an iterator of lists of tasks to perform for each
        batch. All iterators (inner and outer) should feel free to do compute
        intensive operations between returning batches/epochs - they are
        intended to do all training/eval operations with the iterator structure
        allowing for a very modular format.
        """
        raise NotImplementedError


def train(schedule: TrainSchedule): 
    """
    Trains according to a train schedule
    """
    for data_iterator, task_iterator in schedule.epoch_iterator(): 
        for batch, tasks in tqdm(zip(data_iterator, task_iterator), 
                                 total=len(data_iterator), 
                        desc="Batches", dynamic_ncols=True):
            for task in tasks:
                task.process_batch(batch)




