from sacred import Ingredient
from manifold_embedding import ManifoldEmbedding
import random
import time
import torch
import os
import abc
import errno

class Savable(abc.ABC):
    '''
    Abstract class that any savable/loadable model should extend
    '''

    @abc.abstractmethod
    def get_save_data(self):
        '''
        Gets all necessary data needed to save this object - should be pickelable
        '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_save_data(data, cls):
        '''
        Should return an instance of the object implementing this abstract class using the data returned from get_save_data
        ''' 
        raise NotImplementedError


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
def save_model(model: Savable, extra_data, tries, path, _log):
    '''
    Saves a Savable model in addition to any extra data
    '''
    params = {
        'model': model.get_save_data(),
        'model_class': model.__class__,
        'extra_data': extra_data
    }
    return save_data(path, params, tries, _log)

@save_ingredient.capture
def save_data(path, data, tries, _log):
    '''
    Saves data to be pickeled at path
    '''
    try:
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save(data, path)
        return path
    except Exception as err:
        if tries > 0:
            _log.warning(f'Exception while saving ({err})\nRetrying ({tries})')
            time.sleep(60)
            save_data(data, tries=(tries - 1))
        else:
            _log.warning("Giving up on saving...")


@save_ingredient.capture
def load(path):
    '''
    Loads pickeled data
    '''
    return torch.load(path, map_location='cpu')


@save_ingredient.capture
def load_model(path):
    '''
    Returns Savable model, extra_data as saved in the above save function
    '''
    params = load(path)
    model = params['model_class'].from_save_data(params['model'])
    model.to(torch.device('cpu'))
    return model, params['extra_data']
    
