import sys
import threading

from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from stardist_tools import fill_label_holes
from stardist_tools.csbdeep_utils import normalize

from stardist_tools.sample_patches import get_valid_inds, sample_patches
from stardist_tools.utils import edt_prob, mask_to_categorical
from stardist_tools.geometry import star_dist3D, star_dist

from stardist_tools.nms import _ind_prob_thresh
from stardist_tools.utils import _is_power_of_2, optimize_threshold
from scipy.ndimage import zoom

from .networks import define_stardist_net, DistLoss
from .logger import Logger
from .utils import with_no_grad, get_scheduler, update_lr, makedirs, _make_grid_divisible, load_json, save_json, \
    makedirs, load_img, TimeTracker
from .transforms import get_params, get_transforms

from stardist_tools.rays3d import Rays_GoldenSpiral, rays_from_json

from .config import Config3D


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
        # self.rays=rays
        self.image_ndim = len(opt.kernel_size)

        self.n_channel = opt.n_channel
        self.sd_mode = 'opencl' if self.opt.use_gpu else 'cpp'

        self.grid = tuple(opt.grid)
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in opt.grid)
        self.anisotropy = opt.anisotropy

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        n_images = len(image_paths)
        n_masks = len(mask_paths)
        assert len(image_paths) == len(
            mask_paths), f"The nb of image paths, {n_images}, is different of the nb of mask paths, {n_masks}"

        if augmenter is None:
            augmenter = lambda *args: args
        assert callable(augmenter), "augmenter must be None or callable."
        self.augmenter = augmenter

        if opt.use_opencl:
            from gputools import max_filter
            self.max_filter = lambda y, patch_size: max_filter(y.astype(np.float32), patch_size)
        else:
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
            return self.get_valid_inds(mask, foreground_prob=0)
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
            image = load_img(image_path).squeeze().astype("float32")
            mask = load_img(mask_path).squeeze()

            mask_int = mask.astype(np.uint16)
            if (mask_int != mask).any():
                mask_int = mask_int = mask.astype(np.uint32)
                if (mask_int != mask).any():
                    warnings.warn(f" mask <{mask_path}> cannot be converted to np.uint32 whithout losing informations!")
            del mask
            mask = mask_int

            if normalize_channel != "none":

                if self.image_ndim == 3:
                    axis_norm = (-1, -2, -3) if normalize_channel == "independently" else (-1, -2, -3, -4)
                else:
                    axis_norm = (-1, -2) if normalize_channel == "independently" else (-1, -2, -3)

                image = normalize(image, 1, 99.8, axis=axis_norm)  # , clip=True)

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

        assert ndim in (2, 3), f"len(patch_size={patch_size}) not in (2, 3)"
        assert image.ndim in (ndim, ndim + 1), f"image.ndim not in ({(ndim, ndim + 1)}). image.shape={image.shape}"
        assert mask.ndim == ndim, f"mask.ndim != {ndim}. mask.shape={mask.shape}"
        assert image.shape[-ndim:] == mask.shape[
                                      -ndim:], f"images and masks should have corresponding shapes/dimensions. Found image.shape={image.shape} and mask.shape={mask.shape}"

        # mask = mask[np.newaxis]

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

        # Augmentation works with 3D images, so if the image is in 2D, convert it 3D by adding an additional dim
        if self.image_ndim == 2:
            patch_size = [1] + list(patch_size)
            image = image[np.newaxis]
            mask = mask[np.newaxis]

        transform_param = get_params(self.opt, patch_size)
        transform_image = get_transforms(self.opt, transform_param, is_mask=False)
        transform_mask = get_transforms(self.opt, transform_param, is_mask=True)

        image = transform_image(image)
        mask = transform_mask(mask)

        # If image in 2D, remove dimension added before augmentation
        if self.image_ndim == 2:
            patch_size = patch_size[1:]
            image = image[0]
            mask = mask[0]

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
        if self.image_ndim == 3:
            dist = star_dist3D(mask, self.rays, mode=self.sd_mode, grid=self.grid)
        else:
            dist = star_dist(mask, self.n_rays, mode=self.sd_mode, grid=self.grid)
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


class StarDistBase(nn.Module):

    def __init__(self, opt, rays=None):
        super().__init__()

        self.thresholds = dict(
            prob=0.5,
            nms=0.4,
        )

        # if opt.rays_json['kwargs']["anisotropy"] is None:
        #     warnings.warn("Anisotropy was not set. Assuming isotropy; this may reduce performances. if it is not the case")

        self.opt = opt
        self.opt.n_dim = len(self.opt.kernel_size)
        self.isTrain = opt.isTrain
        self.use_amp = opt.use_amp if hasattr(opt, "use_amp") else False

        if self.use_amp and not opt.use_gpu:
            warnings.warn("GPU is not used (use_gpu=False), so `use_amp` is set to False")
            self.use_amp = False

        self.device = torch.device(f"cuda:0") if opt.use_gpu else torch.device("cpu")
        self.logger = Logger()

        # Define and load networks
        if hasattr(opt, "load_epoch") and opt.load_epoch not in (None, ""):
            self.opt.epoch_count = opt.load_epoch
            name = None
            if opt.load_epoch == "best":
                name = "best"

            load_path = None
            if hasattr(self.opt, "load_path"):
                load_path = opt.load_path
            self.load_state(name=name, load_path=load_path)

            self.opt.n_dim = len(self.opt.kernel_size)

        else:
            if self.opt.n_dim == 3:
                if rays is None:
                    if hasattr(opt, "rays_json"):
                        rays = rays_from_json(opt.rays_json)
                    elif hasattr(opt, 'n_rays'):
                        rays = Rays_GoldenSpiral(opt.n_rays,
                                                 anisotropy=(opt.anisotropy if opt.anisotropy != "auto" else None))
                    else:
                        rays = Rays_GoldenSpiral(96, anisotropy=opt.anisotropy)

                opt.rays_json = rays.to_json()

                if opt.rays_json['kwargs']["anisotropy"] is None:
                    warnings.warn(
                        "Anisotropy was not set. Assuming isotropy; this may reduce performances. if it is not the case")

            self.opt.epoch_count = 0
            self.net = define_stardist_net(opt)

            if self.isTrain:
                self.set_optimizers()
                self.set_criterions()

    def set_optimizers(self):
        opt = self.opt
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.lr_scheduler = get_scheduler(self.optimizer, opt, init_lr=opt.lr)
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def set_criterions(self):
        opt = self.opt
        self.criterion_object = torch.nn.BCEWithLogitsLoss()
        self.criterion_class = torch.nn.CrossEntropyLoss()
        self.criterion_dist = DistLoss(lambda_reg=opt.lambda_reg)

    @with_no_grad
    def evaluate(self, batch, epoch=None):
        opt = self.opt
        if epoch is None:
            epoch = self.opt.epoch_count

        device = self.device

        image = batch['image'].to(device)
        dist = batch['dist'].to(device)
        prob = batch['prob'].to(device)

        prob_class = None
        if "prob_class" in batch:
            prob_class = batch["prob_class"].to(device)
            assert opt.n_classes is not None, f"'prob_class' (type={type(prob_class)}) not  None in batch but opt.n_classes is None"
        else:
            assert opt.n_classes is None, f"'prob_class' is None in batch but opt.n_classes = {opt.n_classes} != None"

        batch_size = image.shape[0]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_dist, pred_prob, pred_prob_class = self.net(image)

            loss_dist = self.criterion_dist(pred_dist, dist, mask=prob)
            loss_prob = self.criterion_object(pred_prob, prob)
            loss_prob_class = torch.tensor(0.)
            if prob_class is not None:
                loss_prob_class = self.criterion_class(pred_prob_class, prob_class)

            loss = loss_prob * opt.lambda_prob + loss_dist * opt.lambda_dist + loss_prob_class * opt.lambda_prob_class

        self.logger.log("Val_loss", loss.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("Val_loss_prob", loss_prob.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("Val_loss_dist", loss_dist.item(), epoch=epoch, batch_size=batch_size)
        # if prob_class is not None:
        #     self.logger.log("Val_loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("Val_loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)

        return loss.item(), loss_dist.item(), loss_prob.item(), loss_prob_class.item()

    def optimize_parameters(self, batch, epoch=None):

        self.net.train()

        if epoch is None:
            epoch = self.opt.epoch_count

        opt = self.opt
        device = self.device

        image = batch['image'].to(device)
        dist = batch['dist'].to(device)
        prob = batch['prob'].to(device)

        prob_class = None
        if "prob_class" in batch:
            prob_class = batch["prob_class"].to(device)
            assert opt.n_classes is not None, f"'prob_class' (type={type(prob_class)}) not  None in batch but opt.n_classes is None"
        else:
            assert opt.n_classes is None, f"'prob_class' is None in batch but opt.n_classes = {opt.n_classes} != None"

        # if prob_class is not None:
        #    assert opt.n_classes is not None, f"'prob_class' (type={type(prob_class)}) not  None in batch but opt.n_classes is None"
        #    prob_class = prob_class.to(device)
        # else:
        #    assert opt.n_classes is None, f"'prob_class' i None in batch but opt.n_classes = {opt.n_classes} != None"

        batch_size = image.shape[0]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_dist, pred_prob, pred_prob_class = self.net(image)

            loss_dist = self.criterion_dist(pred_dist, dist, mask=prob)
            loss_prob = self.criterion_object(pred_prob, prob)
            loss_prob_class = torch.tensor(0.)
            if prob_class is not None:
                loss_prob_class = self.criterion_class(pred_prob_class, prob_class)

            loss = loss_prob * opt.lambda_prob + loss_dist * opt.lambda_dist + loss_prob_class * opt.lambda_prob_class

        # Mixed Precision ======================
        self.optimizer.zero_grad()
        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()

        self.logger.log("loss", loss.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("loss_prob", loss_prob.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("loss_dist", loss_dist.item(), epoch=epoch, batch_size=batch_size)
        # if prob_class is not None:
        # self.logger.log("loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)
        self.logger.log("loss_prob_class", loss_prob_class.item(), epoch=epoch, batch_size=batch_size)

        return loss.item(), loss_dist.item(), loss_prob.item(), loss_prob_class.item()

    @with_no_grad
    def predict(self, image, patch_size=None, context=None):
        """
        parameters
        ----------
        image : np.ndarray
            image or volume. shape = (channel, height, width) if self.n_dim==2 and = (channel, depth, height, width) if self.ndim==3
        """

        if patch_size is not None and context is not None:
            return self.predict_big(image, patch_size=patch_size, context=context)
        else:
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)
            pred_dist, pred_prob, pred_prob_class = self.net.predict(image)

        pred_dist = np.moveaxis(pred_dist.cpu().numpy(), 1, -1)
        pred_prob = np.moveaxis(pred_prob.cpu().numpy(), 1, -1)
        if pred_prob_class is not None:
            pred_prob_class = np.moveaxis(pred_prob_class.cpu().numpy(), 1, -1)

        return pred_dist[0], pred_prob[0], (None if pred_prob_class is None else pred_prob_class[0])

    @with_no_grad
    def predict_sparse(self, image, b=2, prob_thresh=None, patch_size=None, context=None):
        if prob_thresh is None:
            prob_thresh = self.thresholds["prob"]

        if patch_size is not None and context is not None:
            print(" === Per patch inference")
            # dist, prob, prob_class = self.predict_big(image, patch_size=patch_size, context=context)
        # else:
        #     dist, prob, prob_class = self.predict(image)
        dist, prob, prob_class = self.predict(image, patch_size=patch_size, context=context)

        # ...
        prob = prob[..., 0]
        # dist = np.moveaxis(dist, 0, -1)
        dist = np.maximum(1e-3, dist)
        #
        inds = _ind_prob_thresh(prob, prob_thresh, b=b)

        prob = prob[inds].copy()
        dist = dist[inds].copy()
        points = np.stack(np.where(inds), axis=1)
        points = points * np.array(self.opt.grid).reshape(1, len(self.opt.grid))

        if self._is_multiclass():
            assert prob_class is not None, f"prediction 'prob_class' is None but self.is_multiclass()==True"
            # prob_class = np.moveaxis(prob_class, 0, -1)
            prob_class = prob_class[inds].copy()

        return dist, prob, prob_class, points

    def _prepare_patchsize_context(self, patch_size, context):
        if context is None:
            if not hasattr(self, 'receptive_field') or self.receptive_field is None:
                self.receptive_field = [max(rf) for rf in self._compute_receptive_field()]

            context = self.receptive_field
        grid = self.opt.resnet_n_downs
        patch_size = _make_grid_divisible(grid, patch_size, name="patch_size")
        context = _make_grid_divisible(grid, context, name="context")

        print(context, patch_size, grid)

        return patch_size, context

    @with_no_grad
    def predict_instance(
            self,
            image,
            prob_thresh=None,
            nms_thresh=None,
            sparse=True,

            patch_size=None,
            context=None,

            return_label_image=True,
            return_predict=None,
            overlap_label=None,
            predict_kwargs=None, nms_kwargs=None
    ):
        """
        patch_size: tuple of size nb dim of image
            patch size to use for per patch prediction (to avoid OOM)
            default: None -> Perform one pass on the whole image

        context: tuple of size nb dim of image
            size of context to use around each patch during per patch_prediction
            default: None -> Use the model receptive field
        """

        self.net.eval()

        if patch_size is not None:
            patch_size, context = self._prepare_patchsize_context(patch_size, context)

        if predict_kwargs is None:
            predict_kwargs = dict()
        if nms_kwargs is None:
            nms_kwargs = dict()

        if sparse:
            dist, prob, prob_class, points = self.predict_sparse(image, patch_size=patch_size, context=context,
                                                                 **predict_kwargs)

        else:
            dist, prob, prob_class = self.predict(image, patch_size=patch_size, context=context, **predict_kwargs)
            prob = prob[..., 0]  # removing the channel dimension
            points = None
        
        #print('dist shape', dist.shape)
        #print('prob shape', prob.shape)
        
        res = dist, prob, prob_class, points

        # print(dist.shape, prob.shape, prob_class.shape if prob_class is not None else None, points.shape if points is not None else None)

        shape = image.shape[-self.opt.n_dim:]

        res_instances = self._instances_from_prediction(
            shape,
            prob,
            dist,
            points=points,
            prob_class=prob_class,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            return_label_image=return_label_image,
            overlap_label=overlap_label,
            **nms_kwargs
        )

        if return_predict:
            return res_instances, tuple(res[:-1])

        else:
            return res_instances

    def _is_multiclass(self):
        return self.opt.n_classes is not None

    def _compute_receptive_field(self, img_size=None):
        # TODO: good enough?
        if img_size is None:
            img_size = tuple(g * (128 if self.opt.n_dim == 2 else 64) for g in self.opt.grid)
        if np.isscalar(img_size):
            img_size = (img_size,) * self.opt.n_dim
        img_size = tuple(img_size)
        # print(img_size)

        assert all(_is_power_of_2(s) for s in img_size)

        mid = tuple(s // 2 for s in img_size)
        x = np.zeros((1, self.opt.n_channel_in) + img_size, dtype=np.float32)
        z = np.zeros_like(x)
        x[(0, slice(None)) + mid] = 1

        with torch.no_grad():
            y = self.net.forward(torch.from_numpy(x).to(self.device))[0][0, 0].cpu().numpy()
            y0 = self.net.forward(torch.from_numpy(z).to(self.device))[0][0, 0].cpu().numpy()

        grid = tuple((np.array(x.shape[2:]) / np.array(y.shape)).astype(int))
        assert grid == self.opt.grid, (grid, self.opt.grid)
        y = zoom(y, grid, order=0)
        y0 = zoom(y0, grid, order=0)
        ind = np.where(np.abs(y - y0) > 0)
        return [(m - np.min(i), np.max(i) - m) for (m, i) in zip(mid, ind)]

    def update_lr(self, epoch=None, metric=None, metric_name=""):
        if epoch is None:
            epoch = self.opt.epoch_count

        lr = update_lr(self.optimizer, self.lr_scheduler, self.opt, metric=metric)

        self.logger.log("lr", lr, epoch=epoch, batch_size=1)
        if metric is not None:
            self.logger.log(f"{metric_name}_metric", float(metric), epoch=epoch, batch_size=1)

    def save_state(self, name=None):
        save_path = Path(self.opt.checkpoints_dir) / f"{self.opt.name}"
        makedirs(save_path)

        epoch = self.opt.epoch_count

        state = {
            "opt": vars(self.opt),
            "epoch": epoch,

            "model_state_dict": self.net.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "amp_scaler_state_dict": self.amp_scaler.state_dict(),
        }

        if name is None:
            name = f"epoch{epoch}_ckpt"
        torch.save(state, save_path / f"{name}.pth")

        print(f"Networks saved at <{save_path}>")

        log_dir = Path(self.opt.log_dir) / f"{self.opt.name}"
        self.logger.to_pickle(path=log_dir / "metrics_logs.pkl")
        self.logger.to_csv(path=log_dir / "metrics_logs.csv")

        save_json(log_dir / 'last_configuration.json', self.thresholds)

        # with open(log_dir /'last_configuration.yml', 'w') as f:
        #     yaml.dump( vars(self.opt), stream=f )

        print(f"Logger saved at <{log_dir}>")

        self.net.to(self.device)

    def load_state(self, name=None, load_path=None):
        opt = self.opt

        if load_path is None:
            if name is None:
                name = f"epoch{opt.epoch_count}_ckpt"

            load_dir = Path(opt.checkpoints_dir) / f"{opt.name}"
            load_path = load_dir / f"{name}.pth"

        print('Load path:', load_path, self.device)
        state = torch.load(load_path, map_location=str(self.device))

        loaded_opt = state["opt"]
        config_class = Config3D if loaded_opt['n_dim'] == 3 else Config2D

        loaded_opt = config_class(allow_new_params=True, **loaded_opt)
        loaded_opt.epoch_count = state['epoch']
        loaded_opt.name = opt.name
        loaded_opt.use_gpu = opt.use_gpu
        loaded_opt.use_amp = opt.use_amp
        loaded_opt.checkpoints_dir = opt.checkpoints_dir
        loaded_opt.log_dir = opt.log_dir

        if opt.n_epochs > loaded_opt.n_epochs:
            loaded_opt.n_epochs = opt.n_epochs
        self.opt = loaded_opt

        if self.opt.n_dim == 3:
            if self.opt.rays_json['kwargs']["anisotropy"] is None:
                warnings.warn(
                    "Anisotropy is not in the checkpoint. Assuming isotropy; This may reduce performances if it is not the case.")

        ################################

        ### Loading thresholds
        checkpoint_dir = Path(self.opt.checkpoints_dir) / f"{self.opt.name}"
        thresholds_path = checkpoint_dir / "thresholds.json"
        if thresholds_path.exists():
            print("Loading threholds ...")
            self.thresholds = load_json(thresholds_path)
        else:
            warnings.warn(
                f"Didn't find thresholds in checkpoint at <{thresholds_path}>. Using Default Thresholds: {self.thresholds}")

        print("Instanciating network")
        self.net = define_stardist_net(loaded_opt)
        print(self.net.load_state_dict(state['model_state_dict']))

        if opt.isTrain:

            self.set_optimizers()
            self.set_criterions()

            if not (hasattr(opt, "reset_optimizers") and opt.reset_optimizers):

                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(state['scheduler_state_dict'])

                if "amp_scaler_state_dict" in state:
                    self.amp_scaler.load_state_dict(state['amp_scaler_state_dict'])
                    print("Optimizers, schedulers and amp_scaler loaded.")

                else:
                    print("*** amp_scaler not in checkpoint. Initialize a new amp_scaler !!!")
                    print("Optimizers and schedulers loaded.")


            else:
                print(f"opt.reset_optimizers={opt.reset_optimizers}. Optimizers and Schedulers don't loaded.")

            self.logger.load_pickle(Path(opt.log_dir) / f"{opt.name}/metrics_logs.pkl", epoch=self.opt.epoch_count)
            print("Logger loaded.")

        print(f"Loading model from <{load_path}>.\n")

    def optimize_thresholds(
            self,
            X_val,
            Y_val,
            nms_threshs=[0.3, 0.4, 0.5],
            iou_threshs=[0.3, 0.5, 0.7],
            predict_kwargs=None,
            optimize_kwargs=None,
            # save_to_json=True
    ):
        # Modified from https://github.com/stardist/stardist/blob/master/stardist/models/base.py
        """Optimize two thresholds (probability, NMS overlap) necessary for predicting object instances.
        Note that the default thresholds yield good results in many cases, but optimizing
        the thresholds for a particular dataset can further improve performance.
        The optimized thresholds are automatically used for all further predictions
        and also written to the model directory.
        See ``utils.optimize_threshold`` for details and possible choices for ``optimize_kwargs``.
        Parameters
        ----------
        X_val : list of ndarray
            (Validation) input images (must be normalized) to use for threshold tuning.
        Y_val : list of ndarray
            (Validation) label images to use for threshold tuning.
        nms_threshs : list of float
            List of overlap thresholds to be considered for NMS.
            For each value in this list, optimization is run to find a corresponding prob_thresh value.
        iou_threshs : list of float
            List of intersection over union (IOU) thresholds for which
            the (average) matching performance is considered to tune the thresholds.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of this class.
            (If not provided, will guess value for `n_tiles` to prevent out of memory errors.)
        optimize_kwargs: dict
            Keyword arguments for ``utils.optimize_threshold`` function.
        """
        self.net.eval()

        if predict_kwargs is None:
            predict_kwargs = dict()
        else:
            if "patch_size" in predict_kwargs and predict_kwargs["patch_size"] is not None:
                predict_kwargs = deepcopy(predict_kwargs)
                patch_size = predict_kwargs["patch_size"]
                context = predict_kwargs.get("context", None)
                patch_size, context = self._prepare_patchsize_context(patch_size, context)
                predict_kwargs["patch_size"] = patch_size
                predict_kwargs["context"] = context

        if optimize_kwargs is None:
            optimize_kwargs = dict()

        # only take first two elements of predict in case multi class is activated
        # Yhat_val = [self.predict(x, **_predict_kwargs(x))[:2] for x in X_val]

        pred_prob_dist = []
        for x in X_val:
            dist, prob = self.predict(x, **predict_kwargs)[:2]
            dist = np.maximum(1e-3, dist)

            prob = prob[..., 0]  # removing the channel dimension
            pred_prob_dist.append([prob, dist])

        opt_prob_thresh, opt_measure, opt_nms_thresh = None, -np.inf, None

        for _opt_nms_thresh in nms_threshs:

            _opt_prob_thresh, _opt_measure = optimize_threshold(
                Y_val, Yhat=pred_prob_dist,
                model=self,
                nms_thresh=_opt_nms_thresh,
                iou_threshs=iou_threshs,
                **optimize_kwargs
            )

            if _opt_measure > opt_measure:
                opt_prob_thresh, opt_measure, opt_nms_thresh = _opt_prob_thresh, _opt_measure, _opt_nms_thresh

        opt_threshs = dict(prob=opt_prob_thresh, nms=opt_nms_thresh)

        self.thresholds = opt_threshs
        print(end='', file=sys.stderr, flush=True)
        print("Using optimized values: prob_thresh={prob:g}, nms_thresh={nms:g}.".format(prob=self.thresholds["prob"],
                                                                                         nms=self.thresholds["nms"]))

        # log_dir = Path(self.opt.log_dir) / f"{self.opt.name}"
        checkpoint_dir = Path(self.opt.checkpoints_dir) / f"{self.opt.name}"
        dest_path = checkpoint_dir / "thresholds.json"
        makedirs(dest_path)
        print(f"Saving to <{dest_path}>")
        save_json(dest_path, self.thresholds)

        return opt_threshs


class ConfigBase:
    """
        Configuration for a StarDist model.

        Parameters
        ----------
        data_dir: str or None
            path to data directory with the following structure:

            data_dir
                |train
                |----|images
                |----|masks
                |val [Optional]
                |----|images
                |----|masks

            if the `val` directory is absent, the data in the `train` folder will be split.



        patch_size: tuple
            size of image to crop from original images.
        load_epoch: int or 'best' or None
            if not None, will load state corresponding to epoch `load_epoch`

        Attributes
        ----------
        name: str
            Name to give to the model
        random_seed: int
            random seed to use for reproducibility

        log_dir: str
            directory path where to save the logs
        checkpoint_dir: str
            directory where to save model states

        # ========================= dataset ==================================
        val_size: float - default 0.15
            Fraction (0...1) of data from the `train` folder to use as validation set when the `val` folder doesn't exist
        n_rays: int
            Number of rays to use in in the star-convex representation of nuclei shape
        foreground_prob: float between 0 and 1
            Fraction (0..1) of patches that will only be sampled from regions that contain foreground pixels.
        cache_sample_ind: bool
            whether to keep in RAM indices of valid patches
        cache_data: bool
            whether to keep in RAM training data

        batch_size: int
            size of batches
        num_workers: int
            Number of subprocesses to use for data training.

        preprocess: str
            type of augmentation to do on training data.
            available augmentations: none|flip|randintensity|randscale|resize
            you can use muliple augmentation, for example for radnom fliping and random intensity scaling, set: `flip_randintensity`
        preprocess_val: str
            same as preprocess but on validation data
        intensity_factor_range: (float, float):
            range from which to sample weight to multiply image intensities.
            Associated to `randintensity` augmentation.
        intensity_bias_range: (float, float)
            range from which to sample bias to add to image intentsities.
            Associated to `randintensity` augmentation.
        scale_limit: (float, float):
            range from which to sample scale to apply the image.
            Associated to `randscale` augmentation.
        resize_to: tuple
            size to which to resize each image.

        #======================================================================

        # ========================= Training ==================================

        use_gpu: bool
            whether to use GPU
        use_amp: bool
            whether to use Automatic Mixed Precision
        isTrain: bool
            whether to initialize model in traning mode (set optimizers, schedulers ...)
        evaluate: bool
            whether to perform evaluation during traning.


        n_epochs: int
            Number of training epochs
        self.n_steps_per_epoch:
            Number of weights updates per epoch

        lambda_prob: flaot
            Weight for probablitly loss
        lambda_dist:
            Weight for distance loss
        lambda_reg: float
            Regularizer to encourage distance predictions on background regions to be 0.

        start_saving_best_after_epoch: int
            Epoch after which to start to save the best model


        #======================================================================



        # ========================= Networks configurations ==================
        grid: str
            Subsampling factors (must be powers of 2) for each of the axes.
            Model will predict on a subsampled grid for increased efficiency and larger field of view.

        n_channel_in: int
            Number of channel of images
        kernel_size: tuple
            Kernel size to use in neural network

        resnet_n_blocks: int
            Number of ResNet blocks to use
        n_filter_of_conv_after_resnet: int
            Number of filter in the convolution layer before the final prediction layer.
        resnet_n_filter_base: int
            Number of filter to use in the first convolution layer
        resnet_n_conv_per_block: int
            Number of convolution layers to use in each ResNet block.

        #======================================================================


        # ========================= Optimizers ================================
        lr: float
            Learning rate
        lr_policy: str
            learning rate scheduler policy.
            Possible values:
                - "none" -> keep the same learning rate for all epochs
                - "plateau" -> Pytorch ReduceLROnPlateau scheduler
                - "linear_decay" -> linearly decay learning rate from `lr` to 0
                - "linear" -> linearly increase  learning rate from 0 to `lr` during the first `lr_linear_n_epochs` and use `lr` for the remaining epochs
                - "step" -> reduce learning rate by 10 every `lr_step_n_epochs`
                - "cosine" -> Pytorch CosineAnnealingLR

        | Parameter for ReduceLROnPlateau used when `lr_policy` = "plateau"
        ------------------------------------------------------------------
        | lr_plateau_factor: float
        | lr_plateau_threshold: float
        | lr_plateau_patience: float
        | min_lr:float
        ------------------------------------------------------------------

        self.lr_linear_n_epochs : int
            See `lr_policy` when `lr_policy`="linear"
        self.lr_step_n_epochs: int
            See `lr_policy` when `lr_policy`="step"
        self.T_max: int
            T_max parameter of Pytorch CosineAnnealingLR.
            Used when `lr_policy` = "cosine

    """

    def update_params(self, allow_new_params=False, **kwargs):
        return self._update_params_from_dict(allow_new_params, kwargs)

    def _update_params_from_dict(self, allow_new_params, param_dict):
        if not allow_new_params:
            attr_new = []
            for k in param_dict:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in param_dict:
            setattr(self, k, param_dict[k])

    def get_params_value(self):
        return self.__dict__