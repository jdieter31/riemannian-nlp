from embed_experiment import ex

ex.run("train", config_updates={"manifold_name": "Poincare"})
ex.run("train", config_updates={"manifold_name": "Euclidean"})
ex.run("train")