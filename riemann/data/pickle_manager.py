import os
import pickle

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
pickle_dir = os.path.join(root_path, "data/pickles/")

def load_or_gen(data_name, gen_func):
    """
    Loads or generates and saves a piece of data

    Params:
        data_name (str): Unique name for this piece of data for storage
        gen_func (function): Python function to generate the needed piece of
            data if it is not found to load.
    """
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    data_path = os.path.join(pickle_dir, data_name + ".pkl")
    if os.path.exists(data_path):
        data = pickle.load(open(data_path, "rb"))
    else:
        data = gen_func()
        pickle.dump(data, open(data_path, "wb"))

    return data




