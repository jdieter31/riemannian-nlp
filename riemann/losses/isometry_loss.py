from math import log
import logging

import torch
from torch.nn.functional import relu

from ..jacobian import compute_jacobian
from ..manifolds import RiemannianManifold

# Add to ensure there are no anomalous zero eigenvalues
EPSILON = 1e-8


logger = logging.getLogger(__name__)


def proximity_loss(model, input_embeddings: torch.Tensor):
    """
    Constructs an isometry.

    Parameters:
        model: model that takes in embeddings in original space and outputs embeddings in output
               manifold space
        input_embeddings (torch.Tensor): tensor of shape [batch_size,
            embedding_sim] with embeddings in original space
        in_manifold (RiemannianManifold): RiemannianManifold object
            characterizing original space
        out_manifold (RiemannianManifold): RiemannianManifold object
            characterizing output space
        out_dimension (int): dimension of tensors in out_manifold
        conformality (float): The degree of conformality to use; a value of 1 implies we will use
            a purely isometric loss. A value of 0 uses an unbounded conformal loss.

    Returns:
        pytorch scalar: computed loss
    """
    output_embeddings = model(input_embeddings)
    return torch.mean(torch.norm(output_embeddings - input_embeddings, dim=-1)**2)


def isometry_loss(model, input_embeddings: torch.Tensor, in_manifold:
                  RiemannianManifold, out_manifold: RiemannianManifold,
                  out_dimension: int, conformality: float=1.0):
    """
    See write up for details on this loss functions -- encourages model to be
    isometric or to be conformal
    Parameters:
        model (torch function from tensors of input manifold to tensors of
        output manifold): model that takes in embeddings in original space and
            outputs embeddings in output manifold space 
        input_embeddings (torch.Tensor): tensor of shape [batch_size,
            embedding_sim] with embeddings in original space
        in_manifold (RiemannianManifold): RiemannianManifold object
            characterizing original space
        out_manifold (RiemannianManifold): RiemannianManifold object
            characterizing output space
        out_dimension (int): dimension of tensors in out_manifold
        conformality (float): The degree of conformality to use; a value of 1 implies we will use
            a purely isometric loss. A value of 0 uses an unbounded conformal loss.

    Returns:
        pytorch scalar: computed loss
    """

    jacobian, model_out = compute_jacobian(model, input_embeddings, out_dimension)
    tangent_proj_out = out_manifold.tangent_proj_matrix(model_out)
    jacobian_shape = jacobian.size()
    tangent_proj_out_shape = tangent_proj_out.size()
    tangent_proj_out_batch = tangent_proj_out.view(-1, tangent_proj_out_shape[-2],
                                                   tangent_proj_out_shape[-1])
    jacobian_batch = jacobian.view(-1, jacobian_shape[-2], jacobian_shape[-1])

    tangent_proj_in = in_manifold.tangent_proj_matrix(input_embeddings)
    proj_eigenval, proj_eigenvec = torch.symeig(tangent_proj_in, eigenvectors=True)
    first_nonzero = (proj_eigenval > 1e-3).nonzero()[0][1]
    significant_eigenvec = proj_eigenvec.narrow(-1, first_nonzero,
                                                proj_eigenvec.size()[-1] - first_nonzero)
    significant_eigenvec_shape = significant_eigenvec.size()
    significant_eigenvec_batch = significant_eigenvec.view(-1, significant_eigenvec_shape[-2],
                                                           significant_eigenvec_shape[-1])
    metric_conjugator = torch.bmm(torch.bmm(tangent_proj_out_batch, jacobian_batch),
                                  significant_eigenvec_batch)
    metric_conjugator_t = torch.transpose(metric_conjugator, -2, -1)
    out_metric = out_manifold.get_metric_tensor(model_out)
    out_metric_shape = out_metric.size()
    out_metric_batch = out_metric.view(-1, out_metric_shape[-2], out_metric_shape[-1])
    pullback_metric = torch.bmm(torch.bmm(metric_conjugator_t, out_metric_batch), metric_conjugator)
    in_metric = in_manifold.get_metric_tensor(input_embeddings)
    in_metric_shape = in_metric.size()
    in_metric_batch = in_metric.view(-1, in_metric_shape[-2], in_metric_shape[-1])
    sig_eig_t = torch.transpose(significant_eigenvec_batch, -2, -1)
    in_metric_reduced = torch.bmm(torch.bmm(sig_eig_t, in_metric_batch), significant_eigenvec_batch)

    # We'll regularize the pullback metric a wee bit.
    n = pullback_metric.shape[-1]
    pullback_metric += EPSILON * torch.eye(n)
    rd = conformal_divergence(pullback_metric, in_metric_reduced, conformality)
    loss = rd.mean()

    return loss


def riemannian_divergence(matrix_a: torch.Tensor, matrix_b: torch.Tensor):
    """
    Computes the Riemannian distance between two postive definite matrices
    """
    # Add a small positive to get rid of degeneracy
    matrix_a_inv = torch.inverse(matrix_a)
    ainvb = torch.bmm(matrix_a_inv, matrix_b)
    eigenvalues, _ = torch.symeig(ainvb, eigenvectors=True)

    log_eig = torch.log(relu(eigenvalues) + EPSILON)
    # Filter potential nans
    if torch.isnan(log_eig).any():
        logger.warning("Found a nan in divergence score")
        log_eig[torch.isnan(log_eig)] = 0
    if torch.isinf(log_eig).any():
        logger.warning("Found a inf in divergence score")
        log_eig[torch.isinf(log_eig)] = 0
    return (log_eig * log_eig).sum(dim=-1)


def conformal_divergence(matrix_a: torch.Tensor, matrix_b: torch.Tensor,
                         conformality: float = 1.0):
    iso_loss = riemannian_divergence(matrix_a, matrix_b)
    if conformality == 1.0:
        return iso_loss

    log_det_diff = torch.logdet(matrix_a) - torch.logdet(matrix_b)
    conformal_correction = (log_det_diff ** 2) / matrix_a.size(-1)

    if conformality > 0.:
        nlogb2 = matrix_a.size(-1) * (log(conformality) ** 2)
        return torch.where(conformal_correction < nlogb2, iso_loss - conformal_correction, iso_loss)
    else:
        return iso_loss - conformal_correction
