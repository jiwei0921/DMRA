import torch
import torch.nn as nn
import numpy as np


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()





####################################  RGB Network  #####################################
class RGBNet(nn.Module):
    def __init__(self,n_class=2):
        super(RGBNet, self).__init__()

        # original image's size = 256*256*3

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)  # 1/32    4 layers

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)



    def forward(self, x):
        h = x

        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        h1 = h_nopool1              # (256x256)*64

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        h2 = h_nopool2              # (128x128)*128

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h_nopool3 = h
        h = self.pool3(h)
        h3 = h_nopool3              # (64x64)*256

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h_nopool4 = h
        h = self.pool4(h)
        h4 = h_nopool4              # (32x32)*512

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        h5 = h              # (16x16)*512


        return h1,h2,h3,h4,h5



    def copy_params_from_vgg19_bn(self, vgg19_bn):
        features = [
            self.conv1_1, self.bn1_1, self.relu1_1,
            self.conv1_2, self.bn1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.bn2_1, self.relu2_1,
            self.conv2_2, self.bn2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.bn3_1, self.relu3_1,
            self.conv3_2, self.bn3_2, self.relu3_2,
            self.conv3_3, self.bn3_3, self.relu3_3,
            self.conv3_4, self.bn3_4, self.relu3_4,
            self.pool3,
            self.conv4_1, self.bn4_1, self.relu4_1,
            self.conv4_2, self.bn4_2, self.relu4_2,
            self.conv4_3, self.bn4_3, self.relu4_3,
            self.conv4_4, self.bn4_4, self.relu4_4,
            self.pool4,
            self.conv5_1, self.bn5_1, self.relu5_1,
            self.conv5_2, self.bn5_2, self.relu5_2,
            self.conv5_3, self.bn5_3, self.relu5_3,
            self.conv5_4, self.bn5_4, self.relu5_4,
        ]
        for l1, l2 in zip(vgg19_bn.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


####################################  Depth Network  #####################################
class DepthNet(nn.Module):
    def __init__(self, n_class=2):
        super(DepthNet, self).__init__()

        # original image's size = 256*256*3

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)  # 1/32    4 layers

        # depth vector
        self.conv_fcn2 = nn.Conv2d(512, 64, 3, padding=1)
        self.pool_avg = nn.AvgPool2d(16, stride=2, ceil_mode=True)
        self.conv_c = nn.Conv2d(64, 6, 1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x

        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        d1 = h_nopool1  # (256x256)*64

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        d2 = h_nopool2  # (128x128)*128

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h_nopool3 = h
        h = self.pool3(h)
        d3 = h_nopool3  # (64x64)*256

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h_nopool4 = h
        h = self.pool4(h)
        d4 = h_nopool4  # (32x32)*512

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        d5 = h          # (16x16)*512

        # depth vector
        vector = self.conv_fcn2(d5)
        vector = self.pool_avg(vector)
        depth_vector = self.conv_c(vector)



        return depth_vector, d1, d2, d3, d4, d5

    def copy_params_from_vgg19_bn(self, vgg19_bn):
        features = [
            self.conv1_1, self.bn1_1, self.relu1_1,
            self.conv1_2, self.bn1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.bn2_1, self.relu2_1,
            self.conv2_2, self.bn2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.bn3_1, self.relu3_1,
            self.conv3_2, self.bn3_2, self.relu3_2,
            self.conv3_3, self.bn3_3, self.relu3_3,
            self.conv3_4, self.bn3_4, self.relu3_4,
            self.pool3,
            self.conv4_1, self.bn4_1, self.relu4_1,
            self.conv4_2, self.bn4_2, self.relu4_2,
            self.conv4_3, self.bn4_3, self.relu4_3,
            self.conv4_4, self.bn4_4, self.relu4_4,
            self.pool4,
            self.conv5_1, self.bn5_1, self.relu5_1,
            self.conv5_2, self.bn5_2, self.relu5_2,
            self.conv5_3, self.bn5_3, self.relu5_3,
            self.conv5_4, self.bn5_4, self.relu5_4,
        ]
        for l1, l2 in zip(vgg19_bn.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data








