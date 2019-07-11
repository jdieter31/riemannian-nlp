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
]
lrs = [1e-3,1e-2,1e-1,10]
for lr in lrs:
    for config_update in config_updates:
        config_update = config_update.copy()
        config_update['learning_rate'] = lr
        ex.run("embed", config_updates=config_update)
