from riemann.embed_experiment import ex

config_updates = [
    {
        "embed_manifold_name": "EuclideanManifold",
        "model": {
            "intermediate_manifolds": [
                {
                    "name": "EuclideanManifold",
                    "params": None
                }
            ]
        }
    },

    {
        "embed_manifold_name": "PoincareBall",
        "model": {
            "intermediate_manifolds": [
                {
                    "name": "PoincareBall",
                    "params": None
                }
            ]
        }
    }
]
intermediate_dims = [300, 450, 700, 850, 1000]
for intermediate_dim in intermediate_dims:
    for config_update in config_updates:
        config_update = config_update.copy()
        config_update['model']["intermediate_dims"] = [intermediate_dim]
        config_update['tensorboard_dir'] = f"runs/lr_schedule/{config_update['embed_manifold_name']}/1Layer/hidden_dim{intermediate_dim}"

        ex.run("embed", config_updates=config_update)
