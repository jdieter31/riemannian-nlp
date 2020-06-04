import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from graph_tool import Graph

from . import PoincareBall, RiemannianManifold
from .data.graph_dataset import GraphDataset
from .featurizers.graph_object_id_featurizer_embedder import GraphObjectIDFeaturizerEmbedder
from .manifolds import SphericalManifold


def project_sphere(u, v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z


def project_to_ambient(manifold, data):
    # if isinstance(manifold, SphericalManifold):
    #     assert data.shape[-1] == 2
    #     return np.vstack(project_sphere(data.T[0], data.T[1])).T
    # else:
    return data


def draw_manifold_wireframe(ax, manifold):
    if isinstance(manifold, PoincareBall):
        # (Must be H2)
        # We'll just draw the outer circle
        ax.add_artist(plt.Circle((0, 0), 1., color='b', fill=False))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    if isinstance(manifold, SphericalManifold):
        # (Must be S2)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x, y, z = project_sphere(u, v)
        ax.plot_surface(x, y, z, color="w", alpha=0.2)


def draw_wireframe(ax, inputs, model=None):
    min_x, min_y = np.min(inputs, 0)
    max_x, max_y = np.max(inputs, 0)

    xlinspace = np.linspace(min_x, max_x, 150)
    ylinspace = np.linspace(min_y, max_y, 150)
    rstride, cstride = 10, 10

    wire_in_x, wire_in_y = np.meshgrid(xlinspace, ylinspace)
    wire_in: np.ndarray = np.stack((wire_in_x, wire_in_y), axis=-1)

    wire_out: np.ndarray
    if model:
        wire_in_ = torch.tensor(wire_in, dtype=torch.float,
                               device=next(model.parameters()).device)
        with torch.no_grad():
            wire_out_ = model(wire_in_)
        wire_out = wire_out_.cpu().detach().numpy()
    else:
        wire_out = wire_in

    if wire_out.shape[-1] == 2:
        rstride = np.linspace(0, len(xlinspace)-1, len(xlinspace)//rstride, dtype=int)
        cstride = np.linspace(0, len(ylinspace)-1, len(ylinspace)//cstride, dtype=int)
        ax.plot(wire_out[:, rstride, 0], wire_out[:, rstride, 1],
                color='blue', alpha=0.3)
        wire_out = wire_out.transpose(1, 0, 2)
        ax.plot(wire_out[:, cstride, 0], wire_out[:, cstride, 1],
                color='blue', alpha=0.3)
    else:
        assert wire_out.shape[-1] == 3
        ax.plot_wireframe(wire_out[:, :, 0], wire_out[:, :, 1], wire_out[:, :, 2],
                          rstride=rstride, cstride=cstride, alpha=0.3)


def plot_input(ax, graph_embedder: GraphObjectIDFeaturizerEmbedder, inputs: torch.Tensor):
    # Select the i-th color in the current scheme
    colors = [f'C{i}' for i in range(len(inputs))]
    ax.scatter(inputs.T[0], inputs.T[1], c=colors, alpha=1)
    for id_, (x, y) in zip(graph_embedder.graph_dataset.object_ids, inputs):
        ax.text(x, y, id_,
                horizontalalignment='center', verticalalignment='bottom')

    for edge in graph_embedder.graph_dataset.edges:
        ax.plot(inputs[edge][:, 0], inputs[edge][:, 1], 'm--', linewidth=3, alpha=0.6)
    ax.axis('equal')


def plot_output(ax, graph_dataset: GraphDataset, manifold: RiemannianManifold,
                inputs: np.ndarray, outputs: np.ndarray):
    colors = [f'C{i}' for i in range(len(inputs))]

    if outputs.shape[-1] == 2:
        ax.scatter(outputs.T[0], outputs.T[1], c=colors, alpha=1)
        for id_, (x, y) in zip(graph_dataset.object_ids, outputs):
            ax.text(x, y, id_,
                    horizontalalignment='center', verticalalignment='bottom')

        for edge in graph_dataset.edges:
            # Map edge onto the geodesic
            edge = torch.from_numpy(outputs[edge])
            v = manifold.log(edge[0], edge[1])
            t = torch.from_numpy(np.linspace(0, 1, 100, dtype=np.float32))
            curve = manifold.exp(edge[0], (t * v.reshape(-1, 1)).T).numpy()

            ax.plot(curve[:, 0], curve[:, 1],
                    'm--', linewidth=3, alpha=0.6)
        ax.axis('equal')
    else:
        assert outputs.shape[-1] == 3
        ax.scatter(outputs.T[0], outputs.T[1], outputs.T[2], c=colors, alpha=1)
        for id_, (x, y, z) in zip(graph_dataset.object_ids, outputs):
            ax.text(x, y, z, id_,
                    horizontalalignment='center', verticalalignment='bottom')

        # Map edges onto the geodesic
        for edge in graph_dataset.edges:
            # Map edge onto the geodesic
            edge = torch.from_numpy(outputs[edge])
            v = manifold.log(edge[0], edge[1])
            t = torch.from_numpy(np.linspace(0, 1, 100, dtype=np.float32))
            curve = manifold.exp(edge[0], (t * v.reshape(-1, 1)).T).numpy()

            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                    'm--', linewidth=3, alpha=0.6)
        axisEqual3D(ax)


def plot(graph_embedder: GraphObjectIDFeaturizerEmbedder) -> Figure:
    """
    Visualizes a feature-based embedding of graph data from a manifold into
    another
    """
    in_dimension = graph_embedder.in_dimension
    out_dimension = graph_embedder.out_dimension

    assert in_dimension == 2
    assert out_dimension == 2 or out_dimension == 3

    inputs_tensor = graph_embedder.get_featurizer_graph_embedder().retrieve_nodes(
        graph_embedder.graph_dataset.n_nodes()
    )
    output_tensor = graph_embedder.retrieve_nodes(graph_embedder.graph_dataset.n_nodes())

    inputs = inputs_tensor.detach().numpy()
    outputs = output_tensor.detach().numpy()

    ###
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(121)
    plot_input(ax, graph_embedder, inputs)

    outputs = project_to_ambient(graph_embedder.out_manifold, outputs)
    if outputs.shape[-1] == 2:
        ax = fig.add_subplot(122)
    else:
        assert outputs.shape[-1] == 3
        ax = fig.add_subplot(122, projection='3d')
    draw_manifold_wireframe(ax, graph_embedder.out_manifold)
    draw_wireframe(ax, inputs, graph_embedder.model)
    plot_output(ax, graph_embedder.graph_dataset, graph_embedder.out_manifold, inputs, outputs)

    return fig


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_degree_distribution(graph: Graph, file_name):
    degrees = graph.get_total_degrees(graph.get_vertices())
    plt.hist(degrees)
    plt.savefig(file_name)
    plt.clf()
