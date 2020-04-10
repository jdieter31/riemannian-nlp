# Add to ensure there are no anomalous zero eigenvalues
EPSILON = 0.0001

def isometry_loss(model, input_embeddings: torch.Tensor, in_manifold: RiemannianManifold, out_manifold: RiemannianManifold, out_dimension: int, isometric=False, random_samples=0, random_init = None):
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
        isometric (bool): The function will be optimized to be isometric if
            True, conformal if False. Riemannian distance on the manifold of PD
            matrices is used to optimized the metrics if isometric and cosine
            distance between the flattened metric matrices is used if conformal
        random_samples (int): Number of randomly generated samples to use in
            addition to provided input_embeddings
        random_init (dict): Parameters to use for random generation of samples
        - use format described in manifold_initialization

    Returns:
        pytorch scalar: computed loss
    """
    input_embeddings = model(input_embeddings)
    if random_samples > 0:
        random_samples = torch.empty(random_samples, input_embeddings.size()[1], dtype=input_embeddings.dtype, device=input_embeddings.device)
        initialize_manifold_tensor(random_samples, in_manifold, random_init)
        input_embeddings = torch.cat([input_embeddings, random_samples])

    model = model.embedding_model
    jacobian, model_out = compute_jacobian(model, input_embeddings, out_dimension)
    tangent_proj_out = out_manifold.tangent_proj_matrix(model_out)
    jacobian_shape = jacobian.size()
    tangent_proj_out_shape = tangent_proj_out.size()
    tangent_proj_out_batch = tangent_proj_out.view(-1, tangent_proj_out_shape[-2], tangent_proj_out_shape[-1])
    jacobian_batch = jacobian.view(-1, jacobian_shape[-2], jacobian_shape[-1])

    tangent_proj_in = in_manifold.tangent_proj_matrix(input_embeddings)
    proj_eigenval, proj_eigenvec = torch.symeig(tangent_proj_in, eigenvectors=True)
    first_nonzero = (proj_eigenval > 1e-3).nonzero()[0][1]
    significant_eigenvec = proj_eigenvec.narrow(-1, first_nonzero, proj_eigenvec.size()[-1] - first_nonzero)
    significant_eigenvec_shape = significant_eigenvec.size()
    significant_eigenvec_batch = significant_eigenvec.view(-1, significant_eigenvec_shape[-2], significant_eigenvec_shape[-1])
    metric_conjugator = torch.bmm(torch.bmm(tangent_proj_out_batch, jacobian_batch), significant_eigenvec_batch)
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
    in_metric_flattened = in_metric_batch.view(in_metric_reduced.size()[0], -1)
    pullback_flattened = pullback_metric.view(pullback_metric.size()[0], -1)
    
    if not isometric:
        in_metric_reduced = in_metric_reduced / in_metric_reduced.norm(dim=(-2, -1), keepdim=True)
        pullback_metric = pullback_metric / pullback_metric.norm(dim=(-2, -1), keepdim=True)

    # if isometric:
    rd = riemannian_divergence(in_metric_reduced, pullback_metric)
    rd_scaled = torch.sqrt(rd)
    # rd_scaled = rd
    loss = rd_scaled.mean()

    '''
    else:
        loss = -torch.mean(cosine_similarity(pullback_flattened, in_metric_flattened, -1))
    '''

    return loss

def riemannian_divergence(matrix_a: torch.Tensor, matrix_b: torch.Tensor):
    """
    Computes the Riemannian distance between two postive definite matrices
    """
    matrix_a_inv = torch.inverse(matrix_a)
    ainvb = torch.bmm(matrix_a_inv, matrix_b)
    eigenvalues, _ = torch.symeig(ainvb, eigenvectors=True)
    eigenvalues = relu(eigenvalues)
    log_eig = torch.log(eigenvalues + EPSILON)
    return (log_eig * log_eig).sum(dim=-1)