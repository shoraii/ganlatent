import numpy as np
import collections
import torch


class MeanTracker(object):
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean


class _StreamingMean:
    def __init__(self, val=None, counts=None):
        if val is None:
            self.mean = 0.0
            self.counts = 0
        else:
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            self.mean = val
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1

    def update(self, mean, counts=1):
        if isinstance(mean, torch.Tensor):
            mean = mean.data.cpu().numpy()
        elif isinstance(mean, _StreamingMean):
            mean, counts = mean.mean, mean.counts * counts
        assert counts >= 0
        if counts == 0:
            return
        total = self.counts + counts
        self.mean = self.counts / total * self.mean + counts / total * mean
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.mean, self.counts)
        if isinstance(other, _StreamingMean):
            if other.counts == 0:
                return new
            else:
                new.update(other.mean, other.counts)
        else:
            new.update(other)
        return new


class StreamingMeans(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMean)

    def __setitem__(self, key, value):
        if isinstance(value, _StreamingMean):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, _StreamingMean(value))

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            self[k].update(v)

    def to_dict(self, prefix=""):
        return dict((prefix + k, v.mean) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])