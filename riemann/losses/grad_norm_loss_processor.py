from typing import Callable, List
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
import torch.nn as nn
from ..config.config_loader import get_config
from ..data.batching import BatchTask, DataBatch
import torch

class GradNormLossProcessor(BatchTask):
    """
    Class that handles the computation and optimization of multiple loss
    functions. Supports switching weighting schemes and monitoring gradient
    norms.
    """

    def __init__(self,
                 losses: List[Callable[[DataBatch], torch.Tensor]],
                 optimizer: Optimizer,
                 grad_norm_params: List[nn.Parameter]):
        """
        Params:
            losses: list of functions that take in a DataBatch and output the
                loss as a torch scalar
            optimizer (Optimizer): optimizer used to update the losses 
            grad_norm_params (nn.Parameter): The paramters that are used to
                dynamically measure gradient norms and compute adaptive loss
                weightings
        """
        self.losses = losses
        self.optimizer = optimizer
        self.grad_norm_params = grad_norm_params
        self.gradient_weights = [torch.tensor(torch.FloatTensor([1]),
                                              requires_grad=True) for _ in
                                 self.losses]

        learning_config = get_config().learning
        self.grad_norm_optimizer = Adam(self.gradient_weights,
                                        lr=learning_config.grad_norm_lr)
        self.initial_losses = None
        self.iterations = 0


    def process_batch(self, batch: DataBatch):
        """
        Runs both the grad norm weighting optimizer as well as the main loss
        optimizer on a batch of data

        Params:
            batch (DataBatch): The batch of data to be ran on
        """
        learning_config = get_config().learning
        alpha = learning_config.grad_norm_alpha

        # Save initial loss values if none are saved or it's time to refresh as
        # specified by the config
        save_initial = self.initial_losses is None or self.iterations % \
            learning_config.grad_norm_initial_refresh_rate == 0
        if save_initial:
            self.initial_losses = []

        g_norms = []
        losses = []
        total_loss = 0

        for loss_func, weight in zip(self.losses, self.gradient_weights):
            # Compute all losses and gradient norms
            loss = weight * loss_func(batch)
            if save_initial:
                self.initial_losses.append(loss.detach())
            losses.append(loss)
            g_norms.append(self.compute_grad_norms(loss))
            total_loss += loss_val / len(self.losses)

        if save_initial:
            self.initial_losses = torch.concat(self.initial_losses)

        g_norms = torch.concat(g_norms)
        losses = torch.concat(losses)

        # Compute mathematical quantities from GradNorm paper
        g_norm_avg = g_norms.mean()
        l_hats = losses / self.initial_losses
        l_hat_avg = l_hats.mean()
        inv_rates = l_hats / l_hat_avg

        # GradNorm paper says gradients shouldn't be propogated through the
        # target value
        ideal_grad_norms = (g_norm_avg * (inv_rates ** alpha)).detach()

        # L1 Loss for optimizing loss weightings
        grad_norm_loss = (g_norms - ideal_grad_norms).abs().sum()
        self.grad_norm_optimizer.zero_grad()
        grad_norm_loss.backward()
        self.grad_norm_optimizer.step()

        # Main optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Renormalize loss weights
        with torch.no_grad():
            self.gradient_weights /= sum(self.gradient_weights) \
                / len(self.gradient_weights)

        self.iterations += 1


    def compute_grad_norms(self, loss: torch.Tensor):
        """
        Computes the gradient norms (of self.grad_norm_params) for a loss value
        """
        total_norm = 0
        for param in self.grad_norm_params:
            grad = torch.autograd.grad(loss, param, retain_graph=True,
                                       create_graph=True)
            norm = torch.norm(grad[0], 2)
            total_norm += norm ** 2
        
        total_norm = torch.sqrt(total_norm)
        return total_norm.detach()


                                                                                    

            
        



