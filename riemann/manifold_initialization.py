import torch
import torch.nn.init as init

"""
Utility functions for applying specificic initalizations to parameters on a manifold - automatically does the correct initializations for product manifolds. In both functions the param initializations is a dictionary describing the initializations formatted like so:

    {
        'PoincareBall': {
            'init_func': 'uniform_',
            'params': [-0.001, 0.001]
        },
        'global': {
            'init_func': 'normal_'
        }
    }
init_func can be any funciton from torch.nn.init and params are the respective params for the init_func. When no params are specified, only the tensor is passed to the init_func.
"""

def get_initialized_manifold_tensor(device, dtype, shape, manifold, initializations, requires_grad, project=True):
    tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    initialize_manifold_tensor(tensor, manifold, initializations, project)
    return tensor
    
def initialize_manifold_tensor(tensor, manifold, initializations, project=True):
    manifold_name = manifold.__class__.__name__
    if manifold_name == "ProductManifold":
        for i in range(len(manifold.submanifolds)):
            sub_data = manifold.get_submanifold_value_index(tensor, i)
            initialize_manifold_tensor(sub_data, manifold.submanifolds[i], initializations)
    else:
        initialization = None
        if manifold_name in initializations:
            initialization = initializations[manifold_name]
        elif "global" in initializations:
            initialization = initializations["global"]
        if initialization is None:
            return
        init_func = getattr(init, initialization["init_func"])
        init_func_params = []
        if "params" in initialization:
            init_func_params = initialization["params"]
        args = [tensor] + init_func_params
        init_func(*args)
        if False:
            manifold.proj_(tensor)

            


