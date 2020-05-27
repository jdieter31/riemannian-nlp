import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .featurizers.graph_object_id_featurizer_embedder import GraphObjectIDFeaturizerEmbedder
from .manifolds import SphericalManifold


def plot(graph_embedder: GraphObjectIDFeaturizerEmbedder) -> Figure:
    """
    Visualizes a feature-based embedding of graph data from a manifold into
    another
    """
    in_manifold = graph_embedder.in_manifold
    out_manifold = graph_embedder.out_manifold
    in_dimension = graph_embedder.in_dimension
    out_dimension = graph_embedder.out_dimension

    assert in_dimension == 2
    assert out_dimension == 3 or (in_dimension == 2 and isinstance(out_manifold, SphericalManifold))

    mpl.style.use('seaborn')

    inputs_tensor = graph_embedder.get_featurizer_graph_embedder().retrieve_nodes(
        graph_embedder.graph_dataset.n_nodes()
    )
    output_tensor = graph_embedder.retrieve_nodes(graph_embedder.graph_dataset.n_nodes())

    inputs = inputs_tensor.detach().numpy()
    output = output_tensor.detach().numpy()

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(121)

    np.random.seed(1234324)
    colors = np.random.rand(len(inputs), 3)
    ax.scatter(inputs.T[0], inputs.T[1], c=colors, alpha=1)

    for edge in graph_embedder.graph_dataset.edges:
        ax.plot(inputs[edge][:, 0], inputs[edge][:, 1], 'm--', alpha=0.3)

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(output.T[0], output.T[1], output.T[2], c=colors, alpha=1)

    for edge in graph_embedder.graph_dataset.edges:
        ax.plot(output[edge][:, 0], output[edge][:, 1],
                output[edge][:, 2], 'm--', alpha=0.3)

    min_x = np.min(inputs[:, 0])
    min_y = np.min(inputs[:, 1])
    max_x = np.max(inputs[:, 0])
    max_y = np.max(inputs[:, 1])

    xlinspace = np.linspace(min_x, max_x, 150)
    ylinspace = np.linspace(min_y, max_y, 150)

    wire_in_x, wire_in_y = np.meshgrid(xlinspace, ylinspace)
    wire_in = np.stack((wire_in_x, wire_in_y), axis=-1)
    wire_in = torch.tensor(wire_in, dtype=torch.float,
                           device=next(graph_embedder.model.parameters()).device)

    with torch.no_grad():
        wire_out = graph_embedder.model(wire_in)
    wire_out = wire_out.cpu().detach().numpy()
    ax.plot_wireframe(wire_out[:, :, 0], wire_out[:, :, 1], wire_out[:, :, 2],
                      rstride=10, cstride=10, alpha=0.1)

    if isinstance(out_manifold, SphericalManifold):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color="w", alpha=0.2)

    axisEqual3D(ax)

    return fig


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
