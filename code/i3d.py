from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch.utils.data as data_utl
import numpy as np


# ----------------------------------------#
# 对于MaxPool3dSamePadding结构的定义
# 其为继承nn.MaxPool3d这个方法来进行定义
# ----------------------------------------#
class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):

        (batch, channel, t, h, w) = x.size()
        # ---------------------------------#
        # compute 'same' padding
        # 分别计算维度 t,h以及w的pad
        # ---------------------------------#
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f

        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f

        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        # ----------------------#
        # 将三个维度的pad分别表示
        # 出来之后,将pad求出
        # ----------------------#
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        return super(MaxPool3dSamePadding, self).forward(x)


# --------------------------------------#
# 对于Unit3D这个类的定义
# 我们将Conv3d中的padding设置为0,我们将会
# 根据输入的变化来动态的进行pad
# 对于这个模块我们将其看作2D目标检测中的
# conv+bn+relu
# 即为大结构中一次普通的卷积操作
# --------------------------------------#
class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                 padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride, padding=self.padding, bias=self._use_bias)
        # ------------------------------#
        # 在该类中为use_batch_norm=True
        # 表明将会使用3d的BatchNorm
        # ------------------------------#
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # --------------------------------#
        # 在这个类中其具体的实现顺序为
        # 3d卷积 -> BatchNorm3d -> relu
        # --------------------------------#
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


# -----------------------------------#
# 此为对于InceptionModule的定义
# 为多次卷积堆叠而来的结构
# 将会在InceptionI3d使用
# -----------------------------------#
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')

        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=1,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')

        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=1,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')

        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def forward(self, x):
        # ---------------------------------#
        # 此为根据Inception的结构进行搭建
        # 总共有四个输出,之后将它们堆叠然后输出
        # ---------------------------------#
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))

        return torch.cat([b0, b1, b2, b3], dim=1)


# ------------------------------------------#
# Inception-v1 I3D architecture网络结构定义
# 使用的为上述定义的三个模块
# ------------------------------------------#
class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, dropout_keep_prob=0.5):
        super(InceptionI3d, self).__init__()

        # --------------------------------------#
        # 这个初始化函数的作用为定义在网络结构中会用到的
        # 方法以及属于该类的一些函数的定义
        # --------------------------------------#
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        # -------------------------------#
        # final_endpoint默认为logits
        # -------------------------------#
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        # ------------------------------#
        # 此为定义一个空字典,用于存储模块名称
        # 及其相对应的描述
        # 是键值对的形式
        # ------------------------------#
        self.end_points = {}

        # --------------------------------------------#
        # 该模块的描述为Conv3d_1a_7x7
        # 其所对应的具体形式为Unit3D
        # 当最后一个块为Conv3d_1a_7x7,将会返回相应的结果
        # 以下模块均为如下的定义方式
        # --------------------------------------------#
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.build()
        # -------------------------------------------#
        # 对于logits,我们使用一个3D卷积块来实现即可
        # 这里面的类别是按照 Kinetics数据集的400类来定义的
        # -------------------------------------------#
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    # ---------------------------------------#
    # 当训练我们自己的数据集的时候,可以使用
    # replace_logits来定义自己数据集中的类别数
    # ---------------------------------------#
    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    # ------------------------------------#
    # 在上面的初始化函数中,我们已经将需要使用的
    # 模块全部定义到end_points这个字典里面
    # 这个函数的目的在于将这些模块全部加入至
    # module里面进而方便后续的调用
    # ------------------------------------#
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    # --------------------------------#
    # 整体网络前向传播函数的定义
    # --------------------------------#
    def forward(self, x):
        # ----------------------------------------#
        # 对于之前定义的模块的调用
        # use _modules to work with dataparallel
        # ----------------------------------------#
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        # ------------------------------------------------------------#
        # After passing through all of these modules abovementioned
        # 3d平均池化 -> dropout -> logits(3D卷积块)
        # ------------------------------------------------------------#
        logits = self.logits(self.dropout(self.avg_pool(x)))

        # --------------------------------#
        # 若存在空间压缩,则会进行压缩维度操作
        # 首先在第四维度上压缩,之后再压缩一次
        # 只有当被压缩的维度值为1时压缩才会有效！
        # --------------------------------#
        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)
        # -------------------------------------------------------------------------#
        # 相当于直接将第二维度给合并
        # logits is batch X time X classes, which is what we want to work with
        # 过softmax进行归一化操作得到最后的概率
        # -------------------------------------------------------------------------#
        logits = torch.mean(logits, dim=2)
        return F.softmax(logits)

    # ------------------------------------#
    # 只提取图像特征,并再最后过一层3d平均池化
    # 得到最终的特征图像
    # ------------------------------------#
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)