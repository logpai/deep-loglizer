from CONSTANTS import *


def cut_by_613(instances):
    dev_split = int(0.1 * len(instances))
    train_split = int(0.6 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test
