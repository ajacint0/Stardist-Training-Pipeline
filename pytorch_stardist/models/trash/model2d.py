import numpy as np


from .base import StarDistDataBase, StarDistBase, ConfigBase
from .utils import with_no_grad

from ..nms import non_maximum_suppression, non_maximum_suppression_sparse
from ..geometry import polygons_to_label
from ..matching import relabel_sequential


class StarDistData2D(StarDistDataBase):
    def __init__(
            self,
            opt,
            image_paths=None, mask_paths=None,
            images=None, masks=None,
            augmenter=None
    ):
        super().__init__(opt=opt, image_paths=image_paths, mask_paths=mask_paths, images=images, masks=masks,
                         augmenter=augmenter)

        self.n_rays = opt.n_rays


class StarDist2D(StarDistBase):
    def __init__(self, opt):
        super().__init__(opt)

    @with_no_grad
    def predict_big(self, image, patch_size=(128, 128), context=(64, 64)):
        raise NotImplementedError("Per patch inference for 2D images not supported yet.")

    def _instances_from_prediction(
            self,
            img_shape,
            prob,
            dist,
            points=None,
            prob_class=None,
            prob_thresh=None,
            nms_thresh=None,
            return_label_image=True,
            overlap_label=None,
            **nms_kwargs
    ):

        self.net.eval()

        if prob_thresh is None:
            prob_thresh = self.thresholds['prob']
        if nms_thresh is None:
            nms_thresh = self.thresholds['nms']

        # Sparse prediction
        if points is not None:
            points, prob, dist, inds = non_maximum_suppression_sparse(
                dist, prob, points, nms_thresh=nms_thresh, **nms_kwargs
            )
            if prob_class is not None:
                prob_class = prob_class[inds]

        # Dense prediction
        else:
            points, prob, dist = non_maximum_suppression(
                dist, prob, grid=self.opt.grid, prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs
            )
            if prob_class is not None:
                inds = tuple(p // g for p, g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        labels = None

        if return_label_image:
            verbose = nms_kwargs.get('verbose', False)
            verbose and print("render polygons...")

            labels = polygons_to_label(
                dist, points, prob=prob, shape=img_shape, scale_dist=(1, 1)
            )

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label < 0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels)) - {overlap_label}) + 1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _, _ = relabel_sequential(labels)
                # print(np.unique(labels))

        res_dict = dict(dist=dist, points=points,
                        prob=prob)  # , rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict


class Config2D(ConfigBase):
    def __init__(
            self,
            name,
            data_dir=None,
            patch_size=[256, 256],
            load_epoch=None,
            **kwargs
    ):
        super().__init__()

        self.delay_hours = 0.0

        self.name = name  # 'c_elegans_orig'
        self.random_seed = 42

        self.log_dir = './logs/'
        self.checkpoints_dir = "./checkpoints"
        self.result_dir = "./results"

        # ========================= dataset ==================================

        self.data_dir = data_dir  # 'datasets/c_elegans_processed'
        self.val_size = 0.15
        self.n_rays = 32
        self.foreground_prob = 0.9
        self.n_classes = None  # non None value (multiclass) not supported yet
        self.patch_size = patch_size
        self.cache_sample_ind = True
        self.cache_data = True

        self.batch_size = 4
        self.num_workers = 0

        self.preprocess = "none"
        self.preprocess_val = "none"
        self.intensity_factor_range = [0.6, 2.]
        self.intensity_bias_range = [-0.2, 0.2]
        self.scale_limit = [1., 1.1]
        self.resize_to = [5, 286, 286]
        self.crop_size = [1, 256, 256]

        # ======================================================================

        # ========================= Training ==================================

        self.use_gpu = True
        self.use_amp = True
        self.isTrain = True
        self.evaluate = True  # True
        # self.gpu_ids                       = [0]
        # self.continue_train                = False

        self.load_epoch = load_epoch
        self.n_epochs = 400
        self.n_steps_per_epoch = 100

        self.lambda_prob = 1.
        self.lambda_dist = 0.2
        self.lambda_reg = 0.0001
        self.lambda_prob_class = 1.

        self.save_epoch_freq = 50
        self.start_saving_best_after_epoch = 50

        # ======================================================================

        # ========================= Networks configurations ==================
        self.init_type = "normal"
        self.init_gain = 0.02

        self.backbone = "resnet"
        self.grid = "auto"
        self.anisotropy = "auto"

        self.n_channel_in = 1
        self.kernel_size = [3, 3]
        self.resnet_n_blocks = 3
        self.resnet_n_downs = None  # WILL BE SET to 'grid' in the code
        self.n_filter_of_conv_after_resnet = 128
        self.resnet_n_filter_base = 32
        self.resnet_n_conv_per_block = 3
        self.use_batch_norm = False

        # unet configurations

        self.unet_n_depth = 3
        self.unet_kernel_size = 3, 3
        self.unet_n_filter_base = 32
        self.unet_n_conv_per_depth = 2
        self.unet_pool = 2, 2
        self.unet_batch_norm = False
        self.unet_dropout = 0.0
        self.net_conv_after_unet = 128

        # ======================================================================

        # ========================= Optimizers ================================
        self.lr = 0.0003
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.lr_policy = "plateau"
        self.lr_plateau_factor = 0.5
        self.lr_plateau_threshold = 0.0000001
        self.lr_plateau_patience = 40
        self.min_lr = 1e-6

        self.lr_linear_n_epochs = 100
        self.lr_decay_iters = 100
        self.T_max = 2

        self.update_params(**kwargs)

