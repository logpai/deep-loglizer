import torch
import random
import os
import numpy as np
import h5py
import json
import pickle


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def tensor2flatten_arr(tensor):
    return tensor.data.cpu().numpy().reshape(-1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(device=-1):
    if device != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(device))
    else:
        device = torch.device("cpu")
    return device


# def dict2hdf5(filename, dic):
#     with h5py.File(filename, 'w') as h5file:
#         recursive_dict2hdf5(h5file, '/', dic)


# def recursive_dict2hdf5(h5file, path, dic):
#     for key, item in dic.items():
#         if not isinstance(key, str):
#             key = str(key)
#         if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
#             h5file[path + key] = item
#         elif isinstance(item, list):
#             h5file[path + key] = np.array(item)
#         elif isinstance(item, dict):
#             recursive_dict2hdf5(h5file, path + key + '/',
#                                 item)
#         else:
#             raise ValueError('Cannot save %s type' % type(item))


def dump_pickle(obj, file_path):
    print("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    print("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, "/", dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file, "/")


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans