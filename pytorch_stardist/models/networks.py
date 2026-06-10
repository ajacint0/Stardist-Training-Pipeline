import itertools
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch.nn as nn
import torch
from torch.nn import init

from . import convnext
from monai.networks.nets.basic_unet import UpCat

######################################################################
#                    BaseNetwork and helper
######################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


class BaseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def print_network(self):
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_net(self, init_type='normal', init_gain=0.02):
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
        """Initialize a network:
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
        Return an initialized network.
        """
        self.init_weights(init_type, init_gain=init_gain)

    def init_weights(self, init_type='normal', init_gain=0.02):
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/models/networks.py
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        #print('initialize network with %s' % init_type)
        self.apply(init_func)  # apply the initialization function <init_func>
        # return net

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, init_gain)


def define_stardist_net(opt):
    if opt.backbone not in {"resnet", "unet", "convnext_unet"}:
        raise NotImplementedError(f"<{opt.backbone}> is not supported. Backbone supported: <unet>  <convnext_unet>")

    if opt.backbone == "unet":
        net = StarDistUnet(config=opt)
    elif opt.backbone == "convnext_unet":
        net = StarDistConvnextUnet(config=opt)

    net.init_net(init_type=opt.init_type, init_gain=opt.init_gain)
    #net.print_network()

    return net


######################################################################
#                              UNet
######################################################################


def conv_block2(out_channels, n1, n2, activation=nn.ReLU, padding='same', dropout=0.0, batch_norm=False, in_channels=3):
    """
    Create a 2D convolution layer.

    Parameters
    ----------
    out_channels: int
        number of output channels
    n1: int
        kernel size x direction
    n2: int
        kernel size y direction
    activation: nn.Module | Callable
        activation function
    padding: str
        padding option
    dropout: float
        dropout probability
    batch_norm: bool
        if True uses Batch norm
    in_channels: int
        number of input channels

    Returns
    -------
    Sequential
        Returns a :class:`nn.Sequential` object containing the layers for the 2D convolution
    """
    layers = [nn.Conv2d(in_channels, out_channels, (n1, n2), padding=padding)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def conv_block3(out_channels, n1, n2, n3, activation=nn.ReLU, padding='same', dropout=0.0, batch_norm=False, in_channels=3):
    """
    Create a 3D convolution layer.

    Parameters
    ----------
    out_channels: int
        number of output channels
    n1: int
        kernel size x direction
    n2: int
        kernel size y direction
    n3: int
        kernel size z direction
    activation: nn.Module
        activation function
    padding: str
        padding option
    dropout: float
        dropout probability
    batch_norm: bool
        if True uses Batch norm
    in_channels: int
        number of input channels

    Returns
    -------
    Sequential
        Returns a :class:`nn.Sequential` object containing the layers for the 3D convolution
    """
    layers = [nn.Conv3d(in_channels, out_channels, (n1, n2, n3), padding=padding)]
    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    layers.append(activation())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, n_depth=2, n_filter_base=16, kernel_size=(3, 3, 3), n_conv_per_depth=2,
                 activation=nn.ReLU, batch_norm=False, dropout=0.0, last_activation=None, pool=(2, 2, 2),
                 expansion=2):
        super().__init__()
        if len(pool) != len(kernel_size):
            raise ValueError('kernel and pool sizes must match.')

        dim = len(kernel_size)
        if dim != 3:
            raise ValueError('unet_block only 3d.')

        self.n_depth = n_depth

        conv_block = conv_block3
        self.pooling = nn.MaxPool3d(pool)
        
        self.upsampling = nn.Upsample(scale_factor=tuple(pool), mode='nearest')

        self.down_convs = nn.ModuleList()
        self.middle_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        if last_activation is None:
            last_activation = activation

        # down...
        for n in range(n_depth):
            down_convs_n = []
            for i in range(n_conv_per_depth):
                layer = conv_block(int(n_filter_base * expansion ** n), *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   in_channels=in_channels)
                down_convs_n.append(layer)
                in_channels = int(n_filter_base * expansion ** n)

            self.down_convs.append(nn.Sequential(*down_convs_n))

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(int(n_filter_base * expansion ** n_depth), *kernel_size,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm,
                               in_channels=in_channels)
            in_channels = int(n_filter_base * expansion ** n_depth)
            self.middle_convs.append(layer)

        self.middle_convs.append(
            conv_block(int(n_filter_base * expansion ** max(0, n_depth - 1)), *kernel_size,
                       dropout=dropout,
                       activation=activation,
                       batch_norm=batch_norm,
                       in_channels=in_channels)
        )
        in_channels = int(n_filter_base * expansion ** max(0, n_depth - 1))

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            up_convs_n = []
            # add skip layer channel num to total in_channels
            in_channels += int(n_filter_base * expansion ** n)
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(int(n_filter_base * expansion ** n), *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   in_channels=in_channels)
                up_convs_n.append(layer)
                in_channels = int(n_filter_base * expansion ** n)

            self.up_convs.append(
                nn.Sequential(*up_convs_n,
                              conv_block(int(n_filter_base * expansion ** max(0, n - 1)), *kernel_size,
                                         dropout=dropout,
                                         activation=activation if n > 0 else last_activation,
                                         batch_norm=batch_norm,
                                         in_channels=in_channels)
                              )
            )
            in_channels = int(n_filter_base * expansion ** max(0, n - 1))

    def forward(self, x):
        skip_layers = []

        #print('down')
        for i, down_conv in enumerate(self.down_convs):
            #print(i, x.shape)
            x = down_conv(x)
            skip_layers.append(x)
            x = self.pooling(x)

        #print('middle')
        for i, middle_conv in enumerate(self.middle_convs):
            #print(i, x.shape)
            x = middle_conv(x)

        #print('up')
        for i, n in enumerate(reversed(range(self.n_depth))):
            #print(i, x.shape)
            x = torch.cat([self.upsampling(x), skip_layers[n]], dim=1)
            x = self.up_convs[-n-1](x)

        return x


class StarDistUnet(BaseNetwork):
    def __init__(self, config: "ConfigBase"):
        super().__init__()
        self.config = config
        unet_kwargs = {k[len('unet_'):]: v for (k, v) in vars(self.config).items() if k.startswith('unet_')}

        pooled = np.array([1] * len(self.config.grid))
        grid_downsampling_layers = []
        in_channels = self.config.n_channel_in
        
        pooling = nn.MaxPool3d
        conv = nn.Conv3d

        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                conv_layer = conv(in_channels, self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                  padding='same')
                activation = nn.ReLU()

                grid_downsampling_layers.append(conv_layer)
                grid_downsampling_layers.append(activation)
                in_channels = self.config.unet_n_filter_base  # change in_channels after the first convolution

            max_pool = pooling(tuple(pool))
            grid_downsampling_layers.append(max_pool)

        self.grid_downsampling = nn.Sequential(*grid_downsampling_layers)

        self.unet_base = UNetBlock(in_channels=in_channels, **unet_kwargs)

        if self.config.net_conv_after_unet > 0:
            self.final_layer = nn.Sequential(
                conv(self.config.unet_n_filter_base, self.config.net_conv_after_unet,
                     self.config.unet_kernel_size, padding="same"),
                nn.ReLU()
            )
            final_layer_channels = self.config.net_conv_after_unet
        else:
            self.final_layer = Identity()
            final_layer_channels = self.config.unet_n_filter_base
        
        kernel_size = tuple([1] * len(self.config.grid))
        self.output_prob = conv(final_layer_channels, 1, kernel_size, padding="same")
        self.output_dist = conv(final_layer_channels, self.config.n_rays, kernel_size, padding="same")

        if self.config.n_classes is not None:
            self.output_prob_classes = nn.Sequential(
                nn.conv(final_layer_channels, self.config.n_classes + 1, kernel_size, padding="same"),
                nn.Softmax()
            )

    def forward(self, x):   
        #print('input', x.shape)
        x = self.grid_downsampling(x)  
        #print('unet input', x.shape)
        x = self.unet_base(x)     
        x = self.final_layer(x)
 
        if self.config.n_classes is not None:
            output_classes = self.output_prob_classes(x)
        else:
            output_classes = None
        
        return self.output_dist(x), self.output_prob(x), output_classes
    
    def predict(self, x):
        rays, prob, class_prob = self.forward(x)
        prob = torch.sigmoid(prob)
        if class_prob is not None:
            class_prob = torch.nn.functional.softmax(class_prob, dim=-(1 + self.n_dim))
        return rays, prob, class_prob

    @staticmethod
    def define_network(config: "ConfigBase") -> "StarDistUnet":
        net = StarDistUnet(config)
        net.init_net(init_type=config.init_type, init_gain=config.init_gain)
        net.print_network()

        return net


######################################################################
#                           ConvNext-UNet
######################################################################

class UNetDecoder(nn.Module):
    """
    UNet Decoder.
    This class refers to `segmentation_models.pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        encoder_channels: number of output channels for all feature maps in encoder.
            `len(encoder_channels)` should be no less than 2.
        decoder_channels: number of output channels for all feature maps in decoder.
            `len(decoder_channels)` should equal to `len(encoder_channels) - 1`.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ratio.
        bias: whether to have a bias term in convolution blocks in this decoder.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        pre_conv: a conv block applied before upsampling.
            Only used in the "nontrainable" or "pixelshuffle" mode.
        interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
            Only used in the "nontrainable" mode.
        align_corners: set the align_corners parameter for upsample. Defaults to True.
            Only used in the "nontrainable" mode.
        is_pad: whether to pad upsampling features to fit the encoder spatial dims.

    """

    def __init__(
        self,
        spatial_dims: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple],
        bias: bool,
        upsample: str,
        pre_conv: Optional[str],
        interp_mode: str,
        align_corners: Optional[bool],
        is_pad: bool,
    ):

        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("the length of `encoder_channels` should be no less than 2.")
        if len(decoder_channels) != len(encoder_channels) - 1:
            raise ValueError("`len(decoder_channels)` should equal to `len(encoder_channels) - 1`.")

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(
                UpCat(
                    spatial_dims=spatial_dims,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                    is_pad=is_pad,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: List[torch.Tensor], skip_connect: int = 3):
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        x = features[0]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)

        return x
    
    
class StarDistConvnextUnet(BaseNetwork):
    def __init__(self, config: "ConfigBase"):
        super().__init__()
        self.config = config
        unet_kwargs = {k[len('unet_'):]: v for (k, v) in vars(self.config).items() if k.startswith('unet_')}
        
        pooled = np.array([1] * len(self.config.grid))
        grid_downsampling_layers = []
        in_channels = self.config.n_channel_in
        
        pooling = nn.MaxPool3d
        conv = nn.Conv3d

        self.encoder = convnext.ConvNeXt(in_chans=config.n_channel_in, 
                                         depths=config.convnext_encoder_depths, 
                                         dims=config.convnext_encoder_channels[1:],
                                         grid=config.grid
                                        )
        
        self.decoder = UNetDecoder(
            spatial_dims=config.convnext_spatial_dims,
            encoder_channels=config.convnext_encoder_channels,
            decoder_channels=config.convnext_decoder_channels,
            act=config.convnext_act,
            norm=config.convnext_norm,
            dropout=config.convnext_dropout,
            bias=config.convnext_decoder_bias,
            upsample=config.convnext_upsample,
            interp_mode=config.convnext_interp_mode,
            pre_conv=config.convnext_pre_conv,
            align_corners=config.convnext_align_corners,
            is_pad=config.convnext_is_pad,
        )
        
        n_filter_base = self.config.convnext_decoder_channels[-1]
        
        if self.config.net_conv_after_unet > 0:
            self.final_layer = nn.Sequential(
                conv(n_filter_base, self.config.net_conv_after_unet,
                     self.config.net_conv_after_unet_kernel_size, padding="same"),
                nn.ReLU()
            )
            final_layer_channels = self.config.net_conv_after_unet
        else:
            self.final_layer = Identity()
            final_layer_channels = self.config.unet_n_filter_base
        
        kernel_size = tuple([2] * len(self.config.grid))
        self.output_prob = conv(final_layer_channels, 1, kernel_size, stride=2)
        self.output_dist = conv(final_layer_channels, self.config.n_rays, kernel_size, stride=2)

        if self.config.n_classes is not None:
            self.output_prob_classes = nn.Sequential(
                nn.conv(final_layer_channels, self.config.n_classes + 1, kernel_size, stride=2),
                nn.Softmax()
            )
        
    def forward(self, x): 
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        final_out = self.final_layer(decoder_out)
        if self.config.n_classes is not None:
            output_classes = self.output_prob_classes(final_out)
        else:
            output_classes = None       
        return self.output_dist(final_out), self.output_prob(final_out), output_classes
    
    def predict(self, x):
        rays, prob, class_prob = self.forward(x)
        prob = torch.sigmoid(prob)
        if class_prob is not None:
            class_prob = torch.nn.functional.softmax(class_prob, dim=-(1 + self.n_dim))
        return rays, prob, class_prob

    @staticmethod
    def define_network(config: "ConfigBase") -> "StarDistConvnextUnet":
        net = StarDistConvnextUnet(config)
        net.init_net(init_type=config.init_type, init_gain=config.init_gain)
        net.print_network()

        return net

    
######################################################################
#                   DIST LOSS
######################################################################

class DistLoss(nn.Module):
    def __init__(self, lambda_reg=0., norm_by_mask=True):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.criterion = nn.L1Loss(reduction="none")
        self.norm_by_mask = norm_by_mask

    def forward(self, input, target, mask=torch.tensor(1.), dim=1, eps=1e-9):
        actual_loss = mask * self.criterion(input, target)
        norm_mask = mask.mean() + eps if self.norm_by_mask else 1
        if self.lambda_reg > 0:
            reg_loss = (1 - mask) * torch.abs(input)

            loss = actual_loss.mean(dim=dim) / norm_mask + self.lambda_reg * reg_loss.mean(dim=dim)

        else:
            loss = actual_loss.mean(dim=dim) / norm_mask
        return loss.mean()
