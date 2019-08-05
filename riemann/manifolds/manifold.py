import abc
import torch

MANIFOLD_TYPES = []

class RiemannianManifold(abc.ABC):
    '''
    Abstract class that any RiemannianManifold object implements. Contains standard functions for Riemannian geometry.
    All methods should allow broadcasting - the last dimension of tensors on the manifold should contain the
    coordinates of the point.
    '''

    @classmethod
    def register_manifold(cls, manifold_class):
        '''
        Register's an implementation of a RiemannianManifold

        Args:
            manifold_class(Class): class extending RiemannianManifold
        '''
        global MANIFOLD_TYPES
        MANIFOLD_TYPES.append(manifold_class)

    @classmethod
    def from_name_params(cls, name, params):
        '''
        Instantiates an instance of manifold with class name {name} via its from_params method
        '''
        for manifold_cls in MANIFOLD_TYPES:
            if manifold_cls.__name__ == name:
                return manifold_cls.from_params(params)
        raise Exception(f"No known manifold with name {name}.")

    @classmethod
    @abc.abstractmethod
    def from_params(cls, params):
        '''Instantiates an instance of this class from a params dictionary

            params(any): Subclass specific parameters - ideally should only contain human readable parameters
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def retr(self, x: torch.Tensor, u: torch.Tensor, indices: torch.Tensor=None):
        '''Performs retraction map on manifold at point x with tangent vector u

        Args:
            x (torch.Tensor): point on the manifold
            u (torch.Tensor): point on tangent space of x
            indices (torch.Tensor): if not None, only projects given indices (in the first dimension) of the tensor

        Returns:
            x_retr (torch.Tensor): retraction map of u at point x
        '''

        raise NotImplementedError

    @abc.abstractmethod
    def retr_(self, x: torch.Tensor, u: torch.Tensor, indices: torch.Tensor=None):
        '''In-place version of retr'''

        raise NotImplementedError

    @abc.abstractmethod
    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim=False):
        '''Computes geodesic distance between two points x and y on the manifold

        Args:
            x (torch.Tensor): first point on the manifold
            y (torch.Tensor): second point on the manifold
            keepdim (Bool): last dimension is not flattened if False

        Returns:
            dist (torch.Tensor): geodesic distance between x and y
        '''

        raise NotImplementedError
    
    @abc.abstractmethod
    def log(self, x: torch.Tensor, y: torch.Tensor):
        '''Computes the log map of a point y from a point x

        Args:
            x (torch.Tensor): the base point of the log map
            y (torch.Tensor): the manifold point to be mapped

        Returns:
            u: (torch.Tensor): a point in the tangent space of x
        '''

        raise NotImplementedError

    @abc.abstractmethod
    def exp(self, x: torch.Tensor, u: torch.Tensor):
        '''Computes the exp map of a tangent vector u from a point x

        Args:
            x (torch.Tensor): the base point of the exp map
            y (torch.Tensor): the tangent vector to be mapped

        Returns:
            u: (torch.Tensor): a point in the tangent space of x
        '''

        raise NotImplementedError

    @abc.abstractmethod
    def proj(self, x: torch.Tensor, indices: torch.Tensor=None):
        '''Projects x onto the manifold

        Args:
            x (torch.Tensor): the point in Euclidean space to be projected
            indices (torch.Tensor): if not None, only projects given indices (in the first dimension) of the tensor

        Returns:
            x_proj (torch.Tensor): the result of the porjection
        '''
        
        raise NotImplementedError

    @abc.abstractmethod
    def proj_(self, x: torch.Tensor, indices: torch.Tensor=None):
        '''In-place version of proj'''

        raise NotImplementedError

    @abc.abstractmethod
    def rgrad(self, x: torch.Tensor, dx: torch.Tensor):
        """Converts a Euclidean gradient into a Riemannian gradient at source point x

        Args:
            x (torch.Tensor): the point the Riemannian gradient is being computed at
            dx (torch.Tensor): the Euclidean gradient to be converted

        Returns:
            dxr (torch.Tensor): the Riemannian gradient

        """
        raise NotImplementedError

    def rgrad_(self, x: torch.Tensor, dx: torch.Tensor):
        """
        In-place version of rgrad where dx will be modified to become the Riemannian gradient
        """
        raise NotImplementedError
