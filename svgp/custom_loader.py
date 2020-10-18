import math
import torch


class BatchDataloader(object):
    def __init__(self, data_x, data_y, batch_size, shuffle=True):
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert self.data_x.size(0) == self.data_y.size(0)
        assert len(self) > 0

    @property
    def size(self):
        return self.data_x.size(0)

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def _sample_batch_indices(self):
        if self.shuffle:
            idx = torch.randperm(self.size)
        else:
            idx = torch.arange(self.size)

        return idx, len(self)

    def __iter__(self):
        idx, n_batches = self._sample_batch_indices()

        for i in range(n_batches):
            _slice = idx[i * self.batch_size: (i + 1) * self.batch_size]
            yield self.data_x[_slice], self.data_y[_slice]
