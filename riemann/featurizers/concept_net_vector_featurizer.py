from .graph_text_featurizer import GraphObjectIDFeaturizer
from .text_featurizer import TextFeaturizer
from ..embedding.conceptnet import standardized_uri
from ..embedding.conceptnet.query import VectorSpaceWrapper
from ..manifolds import RiemannianManifold, EuclideanManifold


class ConceptNetVectorFeaturizer(GraphObjectIDFeaturizer, TextFeaturizer):

    def __init__(self, data_file: str,
                 manifold: RiemannianManifold = EuclideanManifold()):
        """
        Params:
            data_file (str): location of the file containing the vectors in
                conceptnet format
            manifold (RiemannianManifold): manifold to project embeddings onto
                default Euclidean
        """
        self.wrapper = VectorSpaceWrapper(vector_filename=data_file)
        wrapper.load()
        self.manifold = manifold

    def embed_graph_data(self, node_ids: torch.Tensor, object_ids:
    numpy.ndarray) -> torch.Tensor:
        embeddings = []
        for object_id in object_ids:
            vector = self.wrapper.get_vector(object_id)
            embeddings.append(self.manifold.proj(torch.Tensor(vector)))

        return torch.cat(embeddings)

    def embed_text(self, data: List[str]):
        embeddings = []
        for word in data:
            if word.startswith("/c/"):
                vec = self.wrapper.get_vector(word)
            else:
                uri = standardized_uri("en", word)
                vec = wrapper.get_vector(uri)

            oov = np.linalg.norm(vec) == 0
            if oov:
                embeddings.append(None)
            else:
                embeddings.append(self.manifold.proj(torch.Tensor(vec)))

    def get_manifold(self):
        return self.manifold
