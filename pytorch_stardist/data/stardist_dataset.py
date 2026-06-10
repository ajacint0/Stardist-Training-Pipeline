# from __future__ import absolute_import

import threading
from glob import glob
from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

from stardist_tools import fill_label_holes
from stardist_tools.csbdeep_utils import normalize

from stardist_tools.sample_patches import get_valid_inds, sample_patches
from stardist_tools.utils import edt_prob, mask_to_categorical
from stardist_tools.geometry import star_dist3D, star_dist

from .utils import load_img, TimeTracker, seed_worker, get_files
from ..models.transforms import get_params, get_transforms


def get_dataloader(opt, image_paths, mask_paths, rays=None, is_train_loader=True, augmenter=None):
    dataset = StarDistData3D(
        opt=opt,
        image_paths=image_paths, mask_paths=mask_paths,
        rays=rays,
        augmenter=augmenter,
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=is_train_loader
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
        sampler=sampler
    )

    return dataloader


def get_train_val_dataloaders(opt, rays=None, train_augmenter=None):
    train_image_paths = sorted(glob(opt.train_image_paths))
    train_mask_paths = sorted(glob(opt.train_mask_paths))
    val_image_paths = sorted(glob(opt.val_image_paths))
    val_mask_paths = sorted(glob(opt.val_mask_paths))

    
    
    assert len(train_image_paths) > 0 and len(train_mask_paths) > 0 and len(train_image_paths) == len(train_mask_paths)
    assert len(val_image_paths) > 0 and len(val_mask_paths) > 0 and len(val_image_paths) == len(val_mask_paths)

    try:
        val_opt = deepcopy(opt())
    except:
        val_opt = deepcopy(opt)
    assert id(val_opt) != id(opt)

    for attr in ["preprocess_val", "intensity_factor_range_val", "intensity_bias_range_val", "scale_limit_val",
                 "resize_to_val", "crop_size_val"]:
        if hasattr(val_opt, attr):
            setattr(val_opt, attr.replace("_val", ""), getattr(val_opt, attr))

    train_dataloader = get_dataloader(opt, train_image_paths, train_mask_paths, rays, is_train_loader=True, augmenter=train_augmenter)
    val_dataloader = get_dataloader(val_opt, val_image_paths, val_mask_paths, rays, is_train_loader=False)

    train_dataloader.dataset.split = "train"
    val_dataloader.dataset.split = "val"

    return train_dataloader, val_dataloader


class StarDistDataBase(Dataset):
    def __init__(
            self,
            opt,
            image_paths=None, mask_paths=None,
            images=None, masks=None,
            augmenter=None
    ):
        super().__init__()

        if opt.n_classes is not None:
            raise NotImplementedError('Multiclass training not implemented yet')

        self.cache_data = opt.cache_data
        self._data_cache = dict()
        self.cache_sample_ind = opt.cache_sample_ind
        self._ind_cache_fg = {}
        self._ind_cache_all = {}

        if image_paths is None or mask_paths is None:
            assert (images is not None) and (masks is not None)
            assert len(images) == len(masks), f"nb images ({len(images)}) != nb masks ({len(masks)})"

            opt.cache_data = True
            image_paths = ["none"] * len(images)
            mask_paths = ["none"] * len(images)
            self._data_cache = {idx: {"image": image, "mask": mask} for idx, (image, mask) in
                                enumerate(zip(images, masks))}

        self.opt = opt
        self.image_ndim = opt.n_dim

        self.n_channel = opt.n_channel
        self.sd_mode = 'cpp' #'opencl' if self.opt.use_gpu else 'cpp'

        self.grid = tuple(opt.grid)
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in opt.grid)
        self.anisotropy = opt.anisotropy

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        n_images = len(image_paths)
        n_masks = len(mask_paths)
        assert len(image_paths) == len(mask_paths), f"The nb of image paths, {n_images}, is different of the nb of mask paths, {n_masks}"

        if augmenter is None:
            augmenter = lambda *args: args
        assert callable(augmenter), "augmenter must be None or callable."
        self.augmenter = augmenter
        
        
        #print(self.augmenter)
        #print(self.transform)
        

        #if opt.use_opencl:
        #    from gputools import max_filter
        #    self.max_filter = lambda y, patch_size: max_filter(y.astype(np.float32), patch_size)
        #else:
        #    from scipy.ndimage.filters import maximum_filter
        #    self.max_filter = lambda y, patch_size: maximum_filter(y, patch_size, mode='constant')
        from scipy.ndimage.filters import maximum_filter
        self.max_filter = lambda y, patch_size: maximum_filter(y, patch_size, mode='constant')

        self.max_filter_patch_size = opt.patch_size
        self.lock = threading.Lock()

        self.time_tracker = TimeTracker()

    def get_valid_inds(self, k=None, mask=None, patch_size=None, foreground_prob=None):

        max_filter_patch_size = self.max_filter_patch_size
        if max_filter_patch_size is None:
            max_filter_patch_size = patch_size

        if foreground_prob is None:
            foreground_prob = self.opt.foreground_prob
        foreground_only = np.random.uniform() < foreground_prob
        _ind_cache = self._ind_cache_fg if foreground_only else self._ind_cache_all
        if k is not None and k in _ind_cache:
            inds = _ind_cache[k]
        else:
            patch_filter = (lambda y, p: self.max_filter(y, max_filter_patch_size) > 0) if foreground_only else None
            inds = get_valid_inds(mask, patch_size, patch_filter=patch_filter)
            if self.cache_sample_ind:
                with self.lock:
                    _ind_cache[k] = inds
        if foreground_only and len(inds[0]) == 0:
            # no foreground pixels available
            return self.get_valid_inds(k, mask, patch_size, foreground_prob=0)
        return inds

    def channels_as_tuple(self, x):
        pass  # if self.n_channel

    def __len__(self):
        return len(self.image_paths)

    def get_image_mask(self, idx, normalize_channel="independently", apply_transform=False):
        """
            apply_transform: bool
                if true, will apply data augmentation to the mask and the image.
                N.B: in self.__get_item__, data augmentation is done on the patch and not directly on the whole image.
        """

        assert normalize_channel in ("independently", "jointly", "none"), normalize_channel

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if self.cache_data and idx in self._data_cache:
            image = self._data_cache[idx]["image"]
            mask = self._data_cache[idx]["mask"]
        else:
            image = np.load(image_path).astype("float32")
            mask = np.load(mask_path)

            mask_int = mask.astype(np.uint16)
            if (mask_int != mask).any():
                mask_int = mask_int = mask.astype(np.uint8)
                if (mask_int != mask).any():
                    warnings.warn(f" mask <{mask_path}> cannot be converted to np.uint8 whithout losing information!")
            del mask
            mask = mask_int

            if normalize_channel != "none":
                axis_norm = (0, 1, 2)
                image = normalize(image, 2, 98, axis=axis_norm, clip=True)

            mask = fill_label_holes(mask)

            if self.cache_data:
                self._data_cache[idx] = {
                    "image": image,
                    "mask": mask
                }

        patch_size = self.opt.patch_size
        if patch_size is None:
            patch_size = mask.shape[:self.image_ndim]

        ndim = len(patch_size)

        assert ndim == 3, f"len(patch_size={patch_size}) is not 3)"
        assert image.ndim in (ndim, ndim + 1), f"image.ndim not in ({(ndim, ndim + 1)}). image.shape={image.shape}"
        assert mask.ndim == ndim, f"mask.ndim != {ndim}. mask.shape={mask.shape}"
        assert image.shape[-ndim:] == mask.shape[
                                      -ndim:], f"images and masks should have corresponding shapes/dimensions. Found image.shape={image.shape} and mask.shape={mask.shape}"

        if image.ndim == ndim:
            n_channel = 1
            image = image[np.newaxis]
        else:
            channel_dim = - (self.image_ndim + 1)
            n_channel = image.shape[channel_dim]

        assert n_channel == self.n_channel, f"All the image should have opt.n_channel={self.opt.n_channel} channels. Image <{image_path}> has {n_channel} channel(s)"

        if apply_transform is True:
            image, mask = self.transform(image, mask)

        return image, mask

    def get_all_data(self, **kwargs):
        return self.get_n_image_mask(n=-1, **kwargs)

    def get_n_image_mask(self, n=-1, **kwargs):
        if n < 0:
            n = self.__len__()

        X = []
        Y = []

        for idx in range(n):
            image, mask = self.get_image_mask(idx, **kwargs)
            X.append(image)
            Y.append(mask)

        return X, Y

    def channels_as_tuple(self, x):
        if self.n_channel is None:
            return (x,)
        else:
            return tuple(x[i] for i in range(self.n_channel))

    def transform(self, image, mask):
        patch_size = list(mask.shape)

        transform_param = get_params(self.opt, patch_size)
        transform_image = get_transforms(self.opt, transform_param, is_mask=False)
        transform_mask = get_transforms(self.opt, transform_param, is_mask=True)

        image = transform_image(image)
        mask = transform_mask(mask)

        return image, mask

    def __getitem__(self, idx):
        self.time_tracker.tic("loading")

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image, mask = self.get_image_mask(idx)

        channel_dim = - (self.image_ndim + 1)
        n_channel = image.shape[channel_dim]

        patch_size = self.opt.patch_size
        if patch_size is None:
            patch_size = mask.shape

        ndim = len(patch_size)

        self.time_tracker.tac("loading")

        self.time_tracker.tic("valid_inds")
        valid_inds = self.get_valid_inds(k=idx, mask=mask, patch_size=patch_size)
        self.time_tracker.tac("valid_inds")

        self.time_tracker.tic("sampling")

        mask, *image_channels = sample_patches(
            (mask,) + self.channels_as_tuple(image),
            patch_size=patch_size,
            n_samples=1,
            valid_inds=valid_inds
        )
        mask = mask[0]

        image = np.concatenate(image_channels, axis=0)
        assert image.shape[0] == n_channel, image.shape

        assert image.shape[-ndim:] == tuple(patch_size), (image.shape, patch_size)
        assert mask.shape[-ndim:] == tuple(patch_size), (mask.shape, patch_size)
        
        self.time_tracker.tac("sampling")

        self.time_tracker.tic("augmenting")
        image, mask = self.augmenter(image, mask)
        image, mask = self.transform(image, mask)
        self.time_tracker.tac("augmenting")

        self.time_tracker.tic("edt_computing")
        prob = edt_prob(mask, anisotropy=self.anisotropy)[self.ss_grid[1:]]
        self.time_tracker.tac("edt_computing")

        self.time_tracker.tic("distance_computing")
        dist = star_dist3D(mask, self.rays, mode=self.sd_mode, grid=self.grid)
        # print(mask.shape, image.shape, dist.shape)
        dist = np.moveaxis(dist, -1, 0)
        # plt.imshow(dist[0, 0])
        self.time_tracker.tac("distance_computing")

        prob_class = None
        if self.opt.n_classes is not None:
            raise NotImplementedError('Multiclass support not implemented yet')

        item = dict()
        item['image_path'] = image_path.stem if isinstance(image_path, Path) else image_path
        item['mask_path'] = mask_path.stem if isinstance(mask_path, Path) else mask_path
        item['image'] = image.astype("float32")
        item['mask'] = mask.astype("float32")

        item["prob"] = prob[np.newaxis]
        item["dist"] = dist
        if prob_class is not None:
            item["prob_class"] = prob_class

        return item


class StarDistData3D(StarDistDataBase):
    def __init__(
            self,
            opt,
            rays,
            image_paths=None, mask_paths=None,
            images=None, masks=None,
            augmenter=None
    ):
        super().__init__(opt=opt, image_paths=image_paths, mask_paths=mask_paths, images=images, masks=masks,
                         augmenter=augmenter)
        assert rays is not None
        self.rays = rays
