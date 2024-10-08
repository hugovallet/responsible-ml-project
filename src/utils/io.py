import pickle

import yaml


def read_pickle(path, **kwargs):
    with open(path, "rb") as mn:
        return pickle.load(mn, **kwargs)


def read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


# writers with coherent signatures
def write_pickle(obj, path, **kwargs):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, **kwargs)


def write_yaml(obj, path, **kwargs):
    with open(path, "w") as handle:
        yaml.dump(obj, handle, **kwargs)
