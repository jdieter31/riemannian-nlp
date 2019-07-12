from sacred import Ingredient
from manifold_embedding import ManifoldEmbedding
import random
import time
import torch
import os

save_ingredient = Ingredient("save")

@save_ingredient.config
def config():
    path = "models/model"
    i = 1
    while os.path.isfile(path + f"{i}.tch"):
        i += 1
    path += f"{i}.tch"

    tries = 10

@save_ingredient.capture
def save(params, tries, path, _log):
    try:
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save(params, path)
        return path
    except Exception as err:
        if tries > 0:
            _log.warning(f'Exception while saving ({err})\nRetrying ({tries})')
            time.sleep(60)
            save(params, tries=(tries - 1))
        else:
            _log.warning("Giving up on saving...")

@save_ingredient.capture
def load(path, _log):
    params = torch.load(path, map_location='cpu')
    objects = params["objects"]
    objects = params["objects"]
    dimension = params["dimension"]
    double_precision = params["double_precision"]
    manifold = params["manifold"]

    model = ManifoldEmbedding(
        manifold,
        len(objects),
        dimension
    )
    model.to(torch.device('cpu'))
    if double_precision:
        model.double()
    model.load_state_dict(params["model"])
    return model, objects
