from riemann.embed_experiment import ex

config_updates = [
    {
        "embed_manifold_name": "EuclideanManifold",
    },

    {
        "embed_manifold_name": "PoincareBall"
    }
]
lrs = [1e-6,1e-5,1e-4,1e-3,1e-2]
for lr in lrs:
    for config_update in config_updates:
        config_update = config_update.copy()
        config_update['lr_schedule'] = {
            "base_lr": lr,
            "fixed_embedding_lr": lr
        }
        config_update['tensorboard_dir'] = f"runs/{config_update['embed_manifold_name']}/0Layer/lr{lr}/"

        ex.run("embed", config_updates=config_update)
