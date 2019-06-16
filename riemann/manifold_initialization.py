from sacred import Ingredient

import torch.nn.init as init

initialization_ingredient = Ingredient("initialization")



@initialization_ingredient.config
def config():
    global_init = "normal_"
    global_params = []

    manifold_init = {
        "PoincareBall": "uniform_",
    }

    manifold_params = {
        "PoincareBall": [-0.001, 0.001]
    }

def apply_initialization_(tensor, manifold_name, global_init, global_params, manifold_init, manifold_params):
    if manifold_name in manifold_init.keys():
        init_func = getattr(init, manifold_init[manifold_name])
        args = [tensor] + manifold_params[manifold_name]
        init_func(*args)
    else:
        init_func = getattr(init, global_init)
        args = [tensor] + global_params
        init_func(*args)

@initialization_ingredient.capture
def apply_initialization(tensor, manifold, global_init, global_params, manifold_init, manifold_params):
    if manifold.name == "Product":
        for submanifold in manifold.submanifolds:
            sub_data = manifold._get_submanifold_value(tensor, submanifold)
            apply_initialization(sub_data, submanifold, global_init, global_params, manifold_init, manifold_params)
    else:
        apply_initialization_(tensor, manifold.name, global_init, global_params, manifold_init, manifold_params)
