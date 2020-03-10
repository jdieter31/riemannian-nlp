from .graph_embedder import GraphEmbedder
import torch
from torch.nn.functional import relu


def graph_manifold_margin_loss(model: GraphEmbedder, inputs: torch.Tensor,
                                train_distances: torch.Tensor, manifold:
                                RiemannianManifold, margin=0.01,
                                scale_function=lambda x: x):
    """
    See write up for details on this loss function -- encourages embeddings to
    preserve graph topology
    Parameters:
        model (nn.Module): model that takes in graph indices and outputs
            embeddings in output manifold space
        inputs (torch.Tensor): LongTensor of shape [batch_size, num_samples+1]
            giving the indices of the vertices to be trained with the first
            vertex in each element of the batch being the 
            main vertex and the others being samples
        train_distances (torch.Tensor): floating point tensor of shape
            [batch_size, num_samples] containing the training distances from
            the input vertex to the sampled vertices
        manifold (RiemannianManifold): Manifold that model embeds vertices
            into
        scale_function (scalar to scalar function): How manifold distances
            should be scaled for computing loss this could be log, squared,
            identity, or any other increasing function

    Returns:
        pytorch scalar: Computed loss
    """
    
    input_embeddings = model.embed_nodes(inputs)

    # Isolate portion of input that are neighbors
    sample_vertices = input_embeddings.narrow(1, 1,
                                              input_embeddings.size(1) - 1)

    # Isolate portion of input that are main vertices
    main_vertices = input_embeddings.narrow(1, 0,
                                            1).expand_as(sample_vertices)
    manifold_dists = manifold.dist(main_vertices, sample_vertices)

    # Sort neighbors based on given distance
    sorted_indices = train_distances.argsort(dim=-1)
    # View actual manifold distance sorted by given distance
    manifold_dists_sorted = torch.gather(manifold_dists, -1, sorted_indices)
    manifold_dists_sorted = scale_function(manifold_dists_sorted)
    diff_matrix_shape = [manifold_dists.size()[0], manifold_dists.size()[1],
                         manifold_dists.size()[1]]
    row_expanded =
        manifold_dists_sorted.unsqueeze(2).expand(*diff_matrix_shape)
    column_expanded =
        manifold_dists_sorted.unsqueeze(1).expand(*diff_matrix_shape)

    # Produce matrix where the i,j element is d_i - d_j + margin
    # where d_i is the manifold distance from the main vertex to the i'th
    # neighbor
    diff_matrix = row_expanded - column_expanded + margin

    train_dists_sorted = torch.gather(train_distances, -1, sorted_indices)
    train_row_expanded =
        train_dists_sorted.unsqueeze(2).expand(*diff_matrix_shape)
    train_column_expanded =
        train_dists_sorted.unsqueeze(1).expand(*diff_matrix_shape)
    diff_matrix_train = train_row_expanded - train_column_expanded
    # Zero out portions of diff matrix where neighbors are of equal distance
    masked_diff_matrix = torch.where(diff_matrix_train == 0, diff_matrix_train,
                                     diff_matrix)
    masked_diff_matrix = masked_diff_matrix.triu()
    masked_diff_matrix = relu(masked_diff_matrix)
    masked_diff_matrix = masked_diff_matrix.sum(-1)
    loss = masked_diff_matrix.sum(-1).mean()
    return loss


