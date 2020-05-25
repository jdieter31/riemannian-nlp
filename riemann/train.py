import abc
from typing import Callable, cast, Sized
from typing import Iterator, Tuple, List

from tqdm import tqdm

from .data.batching import DataBatch, BatchTask


class TrainSchedule(abc.ABC):
    """
    Abstract class that runs a training operation choosing which dataset to use
    each epoch as well as what tasks to run for each batch of the data. (i.e.
    train one loss, train another, potentially run an evaluation)
    """

    @abc.abstractmethod
    def __init__(self):
        self.iteration_num = 0
        self.epoch_num = 0
        self.cyclic_tasks = []

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

    def run_epoch(self, epoch: Tuple[Iterator[DataBatch],
                                     Iterator[List[BatchTask]]],
                  prog_desc: str = "Batches",
                  count_iterations: bool = True):
        """
        Runs an epoch. This will be called on what is returned from
        epoch_iterator. That should be the main way to start epochs but this
        can be called manually to do things like nested epochs (i.e. run some
        time of batched evaluation midway through training an epoch)

        Params:
            epoch: same as described in epoch_iterator doc
            prog_desc (str): text description for tqdm command line
                progress bar
            count_iterations (bool): Should iterations be counted during this
                epoch? Can be set to False if this is for evaluation rather
                than training.
        """
        data_iterator, task_iterator = epoch
        n_batches = len(cast(Sized, data_iterator))

        for batch, tasks in tqdm(zip(data_iterator, task_iterator),
                                 total=n_batches,
                                 desc=prog_desc, dynamic_ncols=True):
            for task in tasks:
                task.process_batch(batch)

            if count_iterations:
                for task, repeat_every, cycle_on_iterations in \
                        self.cyclic_tasks:

                    if not cycle_on_iterations:
                        # This task is meant for epochs not iterations
                        continue

                    if self.iteration_num % repeat_every == 0:
                        task()

                self.iteration_num += 1

    def add_cyclic_task(self, task: Callable[[], None], repeat_every: int,
                        cycle_on_iterations: bool = True):
        """
        Adds a task to be repeated cyclically during the training cycle.

        Params:
            task (Callable[[], None]): Function to call to do the task
            repeat_every (int): The task will be executed every repeat_every
                iterations (or epochs)
            cycle_on_iterations (bool): If true iterations will be used to
                determine when the task is executed. Otherwise, epochs will be
                used
        """
        self.cyclic_tasks.append((task, repeat_every, cycle_on_iterations))

    def train(self):
        """
        Trains according to the schedule
        """

        for task, repeat_every, cycle_on_iterations in \
                self.cyclic_tasks:

            if cycle_on_iterations:
                # This task is meant for iterations not epochs
                continue

            if self.epoch_num % repeat_every == 0:
                task()

        for epoch in self.epoch_iterator():
            self.run_epoch(epoch)
            self.epoch_num += 1

            for task, repeat_every, cycle_on_iterations in \
                    self.cyclic_tasks:

                if cycle_on_iterations:
                    # This task is meant for iterations not epochs
                    continue

                if self.epoch_num % repeat_every == 0:
                    task()
