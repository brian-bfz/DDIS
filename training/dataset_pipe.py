import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PipeDataset(Dataset):
    """
    Loads 3D turbulent channel flow velocity+pressure snapshots from one or more HDF5 files.
    Each sample is a (4, Nx, Ny, Nz) float32 array: [u, v, w, p], z-score
    normalized per channel across the full dataset.

    `path` may be a single .h5 file path or a glob pattern (e.g.
    "data/oc_full_*/snapshots_north_128.h5") to concatenate multiple runs.
    """

    def __init__(
        self,
        path,
        resolution=None,
        use_labels=False,
        xflip=False,
        cache=False,
        channel=None,
        **kwargs,
    ):
        assert not use_labels, "Labels not supported by PipeDataset"
        assert not xflip, "xflip not supported by PipeDataset"

        self._path = path
        self._name = "pipe"

        files = sorted(glob.glob(path)) if "*" in path else [path]
        if not files:
            raise IOError(f"No HDF5 files found matching: {path}")

        vel_list, pres_list = [], []
        self._coords = None
        self._domain_length = None
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                vel_list.append(f["velocity"][:])   # (T, 3, Nx, Ny, Nz)
                pres_list.append(f["pressure"][:])  # (T, Nx, Ny, Nz)
                if self._coords is None:
                    self._coords = f["coords"][:]            # (3, Nx, Ny, Nz)
                    self._domain_length = f.attrs["domain_length"]

        vel  = np.concatenate(vel_list,  axis=0).astype(np.float32)  # (T, 3, Nx, Ny, Nz)
        pres = np.concatenate(pres_list, axis=0).astype(np.float32)  # (T, Nx, Ny, Nz)

        # (T, 4, Nx, Ny, Nz)
        data = np.concatenate([vel, pres[:, np.newaxis]], axis=1)

        # Per-channel z-score over (T, Nx, Ny, Nz)
        mean = data.mean(axis=(0, 2, 3, 4), keepdims=True)   # (1, 4, 1, 1, 1)
        std  = data.std( axis=(0, 2, 3, 4), keepdims=True).clip(min=1e-6)
        self._data = (data - mean) / std  # float32, (T, 4, Nx, Ny, Nz)

        T, C, Nx, Ny, Nz = self._data.shape

        if resolution is not None:
            assert Nx == Ny == Nz == resolution, (
                f"resolution={resolution} does not match data shape ({Nx},{Ny},{Nz})"
            )

        self._raw_shape = [T, C, Nx, Ny, Nz]

        if channel is None:
            self._channel = None
        elif isinstance(channel, int):
            self._channel = [channel]
        else:
            self._channel = list(channel)

        if self._channel is not None:
            self._raw_shape[1] = len(self._channel)

    def __len__(self):
        return self._raw_shape[0]

    def __getitem__(self, idx):
        sample = self._data[int(idx)]  # (C, Nx, Ny, Nz) float32
        if self._channel is not None:
            sample = sample[self._channel]
        return sample.astype(np.float64), np.zeros(0, dtype=np.float64)

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        """(C, Nx, Ny, Nz)"""
        return list(self._raw_shape[1:])

    @property
    def spatial_shape(self):
        """(Nx, Ny, Nz) — signals to training_loop that this is a 3-D dataset."""
        return tuple(self._raw_shape[2:])

    @property
    def num_channels(self):
        return self._raw_shape[1]

    @property
    def resolution(self):
        Nx, Ny, Nz = self.spatial_shape
        assert Nx == Ny == Nz, (
            f"Non-cubic grid ({Nx},{Ny},{Nz}): use spatial_shape instead of resolution"
        )
        return Nx

    @property
    def label_shape(self):
        return [0]

    @property
    def label_dim(self):
        return 0

    @property
    def has_labels(self):
        return False

    @property
    def has_onehot_labels(self):
        return False
