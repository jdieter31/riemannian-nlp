import importlib
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from math import ceil
from typing import List, Callable, TypeVar, Type, Any, Dict
from zipfile import ZipFile

import torch
from torch.nn import Embedding
from tqdm import tqdm

from .config.config_loader import get_config, initialize_config
from .config.manifold_config import ManifoldConfig
from .data.batching import DataBatch
from .manifold_initialization import initialize_manifold_tensor
from .manifold_tensors import ManifoldParameter
from .manifolds import RiemannianManifold
#: This type is used to represent subclasses of GraphEmbedder in its classmethods
GraphEmbedderType = TypeVar('GraphEmbedderType', bound='GraphEmbedder')


logger = logging.getLogger(__name__)


def dynamically_load_class(path, class_name: str = None) -> Type[Any]:
    """
    Dynamically import a class from @path

    This implementation handles contained classes.

    Args:
        path: A '.' delimited path to a class, e.g., "riemann.graph_embedder.GraphEmbedder"
        class_name: If provided, the class to import from :param:`path`. If not provided, defaults
                    to the last part of :param:`path`.
    """
    if class_name is None:
        path, class_name = path.rsplit('.', 1)
    class_name = class_name.strip()
    try:
        container = importlib.import_module(path)
    except ModuleNotFoundError:
        # Try to see if you can import a subclass.

        path, containing_class = path.rsplit('.', 1)
        container = dynamic_import_class(path, containing_class)

    clazz = getattr(container, class_name)
    if not isinstance(clazz, type):
        raise TypeError(f"{clazz} is not a class.")
    return clazz


class GraphEmbedder(ABC):
    """
    Abstract class for any type of model that produces embeddings of 
    a graph
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def embed_nodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Produces embedding of graph based on input nodes

        Args:
            node_ids (long tensor): input node ids

        Returns:
            embedding (tensor): embedding of nodes
        """
        raise NotImplementedError

    def get_manifold(self) -> RiemannianManifold:
        """
        Returns the manifold that this GraphEmbedder embeds nodes into.
        Defaults to Euclidean if this method is not overwritten
        """

        return ManifoldConfig().get_manifold_instance()

    def get_losses(self) -> List[Callable[[DataBatch], torch.Tensor]]:
        """
        Gets additional losses that should be trained. This is where isometry
        losses are parameter regularization losses should go
        """

        return []

    def retrieve_nodes(self, total_n_nodes):
        """
        Retrieves a matrix of nodes 0 to total_n_nodes on the cpu done in
        batches as specified in the neighbor sampling config 
        """
        sampling_config = get_config().sampling
        num_blocks = ceil(total_n_nodes /
                          sampling_config.manifold_neighbor_block_size)
        block_size = sampling_config.manifold_neighbor_block_size
        out_blocks = []

        for i in tqdm(range(num_blocks), desc=f"Embed {total_n_nodes} Nodes",
                      dynamic_ncols=True):
            start_index = i * block_size
            end_index = min((i + 1) * block_size, total_n_nodes)
            out_blocks.append(self.embed_nodes(torch.arange(start_index,
                                                            end_index,
                                                            dtype=torch.long)).cpu())
        out = torch.cat(out_blocks)
        return out

    # region: serialization
    def _to_file(self, zf: ZipFile) -> None:
        """
        Saves local state (that can't be recovered by re-initializing a model). This typically
        involves torch model state.
        Args:
            zf: A zip-file to write to.
        """
        # Write the actual model class that was used to run the experiment.
        zf.writestr("model.class", f"{self.__class__.__module__}.{self.__class__.__name__}")
        # Save the global configuration so that we can reload this model.
        zf.writestr("config.json", json.dumps(get_config().as_json(), sort_keys=True, indent=2))

    def to_file(self, path: str) -> None:
        """
        Saves this embedder as a .zip file at `path`.

        Args:
            path: the path to write the model to
        """
        # Ensure that the directory exists.
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        with ZipFile(path, "w") as zf:
            self._to_file(zf)

    def _from_file(self, zf: ZipFile) -> Dict[str, Any]:
        """
        Loads state from a saved zip-file.

        Args:
            zf: A zip-file to write to.
        """
        pass

    @classmethod
    def from_file(cls: Type[GraphEmbedderType], path: str, **kwargs) -> GraphEmbedderType:
        """
        Load a model from file, dynamically constructing this class using data stored in zip file.

        Args:
            path: The path to a zipfile to read.
            **kwargs: Any additional arguments to be passed to the constructor.
        """
        # NOTE: lazy loading this method to break a cyclic dependency
        from .model import get_model

        logger.info("Loading model from file: {}".format(path))
        with ZipFile(path, "r") as zf:
            # Initialize global configuration using this file.
            config = json.loads(zf.read("config.json").decode("utf-8"))
            initialize_config(config_updates=config)
            # NOTE[arun]: We get the global model that was set above. Model construction is nearly
            # impossible to isolate because global state is scattered in several places.
            model = get_model()

            # Fill in any state after the fact.
            model._from_file(zf)
        return model
    # endregion


class ManifoldEmbedding(Embedding, GraphEmbedder):
    def __init__(
            self,
            manifold: RiemannianManifold,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            manifold_initialization=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm,
                         norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
                         _weight=_weight)

        self.manifold = manifold
        self.weight = ManifoldParameter(self.weight.data, manifold=manifold)
        if manifold_initialization is not None:
            initialize_manifold_tensor(self.weight.data, self.manifold,
                                       manifold_initialization)

    def get_embedding_matrix(self):
        return self.weight.data

    def get_manifold(self) -> RiemannianManifold:
        return self.manifold

    def embed_nodes(self, node_ids: torch.Tensor):
        node_ids = node_ids.to(self.weight.device)
        return self(node_ids)

    # region: serialization
    def _to_file(self, zf: ZipFile) -> None:
        # Save Torch state to file
        super()._to_file(zf)
        with io.BytesIO() as buf:
            torch.save(self.state_dict(), buf)
            zf.writestr("model.state", buf.getvalue())

    def _from_file(self, zf: ZipFile) -> None:
        # Read Torch state from file
        super()._from_file(zf)
        with io.BytesIO(zf.read("model.state")) as buf:
            # make sure we start by placing things on cpu
            self.load_state_dict(torch.load(buf, map_location="cpu"))
    # endregion
