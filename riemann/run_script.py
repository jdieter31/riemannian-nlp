from embed_experiment import ex

config_updates = [
    {
        "dimension": 20,
        "manifold_name": "PoincareBall"
    },
    {
        "dimension": 20,
        "manifold_name": "Euclidean"
    },
    {
        "dimension": 20,
        "manifold_name": "Sphere"
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "PoincareBall"],
        "submanifold_shapes": [[10], [10]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Euclidean"],
        "submanifold_shapes": [[10], [10]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Sphere"],
        "submanifold_shapes": [[10], [10]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "PoincareBall"],
        "submanifold_shapes": [[15], [5]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Sphere"],
        "submanifold_shapes": [[15], [5]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Euclidean"],
        "submanifold_shapes": [[15], [5]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["Euclidean", "PoincareBall"],
        "submanifold_shapes": [[15], [5]]
    },

    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Euclidean", "Sphere"],
        "submanifold_shapes": [[7], [7], [6]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Euclidean", "PoincareBall"],
        "submanifold_shapes": [[7], [7], [6]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "PoincareBall", "Sphere"],
        "submanifold_shapes": [[7], [7], [6]]
    },
    {
        "dimension": 20,
        "manifold_name": "Product",
        "submanifold_names": ["PoincareBall", "Sphere", "Sphere"],
        "submanifold_shapes": [[8], [6], [6]]
    }
]
for config_update in config_updates:
    ex.run("embed", config_updates=config_update)
