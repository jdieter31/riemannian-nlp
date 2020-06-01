from typing import Callable

import torch
import wandb
from torch.optim.optimizer import Optimizer

from ..data.batching import BatchTask, DataBatch


class SingleLossProcessor(BatchTask):
    """
    Class that handles the computation and optimization of a single loss
    function
    """

    def __init__(self,
                 loss: Callable[[DataBatch], torch.Tensor],
                 optimizer: Optimizer):
        """
        Params:
            loss: function that takes in a DataBatch and outputs the
                loss as a torch scalar
            optimizer (Optimizer): optimizer used to update the losses 
        """
        self.loss = loss
        self.optimizer = optimizer
        self.iterations = 0

    def process_batch(self, batch: DataBatch):
        """
        Runs both the grad norm weighting optimizer as well as the main loss
        optimizer on a batch of data

        Params:
            batch (DataBatch): The batch of data to be ran on
        """
        loss = self.loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        print(f"optimization step {self.iterations}")
        self.optimizer.step()

        wandb.log({"train/loss": float(loss.cpu().detach().numpy())},
                  step=self.iterations)

        self.iterations += 1
