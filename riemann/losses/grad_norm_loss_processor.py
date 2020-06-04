import logging
from typing import Callable, List

import torch
import torch.nn as nn
import wandb
from torch.optim.optimizer import Optimizer

from ..config.config_loader import get_config
from ..data.batching import BatchTask, DataBatch
from ..optimizer_gen import get_scheduler

logger = logging.getLogger(__name__)

EPSILON = 1e-8


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
        self.gradient_weights = torch.tensor([0. for _ in self.losses],
                                             dtype=torch.float32,
                                             requires_grad=False,
                                             device=self.grad_norm_params[0].device)
        self.initial_losses = None

    def process_batch(self, batch: DataBatch, iteration_num: int):
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
        save_initial = (self.initial_losses is None or
                        iteration_num % learning_config.grad_norm_initial_refresh_rate == 0)
        if save_initial:
            self.initial_losses = []

        g_norms = []
        losses = []

        loss_num = 0
        for loss_func in self.losses:
            # Compute all losses and gradient norms
            loss = loss_func(batch)
            if torch.isnan(loss).any():
                logger.warning(f"Loss{loss_num} is NaN")
                # Filter nans
                loss[torch.isnan(loss)] = 0
            wandb.log({f"train/loss{loss_num}": float(loss.cpu().detach().numpy())},
                      step=iteration_num)

            if save_initial:
                self.initial_losses.append(loss.detach())

            g_norm = self.compute_grad_norms(loss).detach()

            losses.append(loss)
            g_norms.append(g_norm)
            wandb.log({f"train/g_norm{loss_num}": float(g_norms[-1].cpu().detach().numpy())},
                      step=iteration_num)
            # total_loss += weight * loss / len(self.losses)
            loss_num += 1

        if save_initial:
            self.initial_losses = torch.stack(self.initial_losses)

        self.optimizer.zero_grad()

        g_norms = torch.stack(g_norms)
        losses = torch.stack(losses)

        # Compute mathematical quantities from GradNorm paper
        g_norm_avg = torch.prod(g_norms) ** (1 / g_norms.size(-1))  # g_norms.mean()
        l_hats = (losses + EPSILON) / (self.initial_losses + losses.size(-1) * EPSILON)
        l_hat_avg = l_hats.mean()
        inv_rates = (l_hats + EPSILON) / (l_hat_avg + EPSILON)

        # GradNorm paper says gradients shouldn't be propogated through the
        # target value
        ideal_grad_norms = (g_norm_avg * (inv_rates ** alpha)).detach()
        self.gradient_weights = ideal_grad_norms / (g_norms + EPSILON)

        self.gradient_weights = (self.gradient_weights.size(0) * self.gradient_weights *
                                 torch.tensor(learning_config.loss_priority,
                                              requires_grad=False,
                                              device=self.gradient_weights.device))
        for i in range(len(self.losses)):
            wandb.log({f"train/g_weight{i}":
                       float(self.gradient_weights[i].cpu().detach().numpy())},
                      step=iteration_num)

        total_loss = (self.gradient_weights * losses).sum()

        total_loss.backward()
        # Main optimization step
        self.optimizer.step()
        # Hacky way of logging learning rate
        for param_group in self.optimizer.param_groups:
            wandb.log({"train/lr": param_group['lr']},
                      step=iteration_num)
            break

        get_scheduler().step(total_loss.detach())

        if any(torch.isnan(param).any() for param in self.grad_norm_params):
            raise RuntimeError("Parameters have been set to NaN")

        wandb.log({f"train/loss_total": float(total_loss.cpu().detach().numpy())},
                  step=iteration_num)

    def compute_grad_norms(self, loss: torch.Tensor):
        """
        Computes the gradient norms (of self.grad_norm_params) for a loss value
        """
        grads = torch.autograd.grad(loss, self.grad_norm_params, retain_graph=True,
                                    create_graph=True)
        if any(torch.isnan(grad).any() for grad in grads):
            logger.warning("Encountered NaN gradient")
        total_norm = torch.sqrt(sum([torch.norm(grad, 2)**2 for grad in grads]))
        return total_norm
