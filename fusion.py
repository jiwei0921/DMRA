import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

'''
fusion: consits of DRB, DMSW, RAM.
'''

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) //2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, 1,
                              self.padding)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, h, c):

        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // self.num_features, dim=1)
        i = torch.sigmoid(ai)    #input gate
        f = torch.sigmoid(af)    #forget gate
        o = torch.sigmoid(ao)    #output
        g = torch.tanh(ag)       #update_Cell

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c, o

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)



        # --------------------------- Depth Refinement Block -------------------------- #
        # DRB 1
        self.conv_refine1_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_refine1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine1_1 = nn.PReLU()
        self.conv_refine1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_refine1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine1_2 = nn.PReLU()
        self.conv_refine1_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_refine1_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine1_3 = nn.PReLU()
        self.down_2_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.down_2_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # DRB 2
        self.conv_refine2_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_refine2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine2_1 = nn.PReLU()
        self.conv_refine2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_refine2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine2_2 = nn.PReLU()
        self.conv_refine2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_refine2_3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine2_3 = nn.PReLU()
        self.conv_r2_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_r2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_r2_1 = nn.PReLU()
        # DRB 3
        self.conv_refine3_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_refine3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine3_1 = nn.PReLU()
        self.conv_refine3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_refine3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine3_2 = nn.PReLU()
        self.conv_refine3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_refine3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine3_3 = nn.PReLU()
        self.conv_r3_1 = nn.Conv2d(256, 64, 3, padding=1)
        self.bn_r3_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_r3_1 = nn.PReLU()
        # DRB 4
        self.conv_refine4_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_refine4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine4_1 = nn.PReLU()
        self.conv_refine4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_refine4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine4_2 = nn.PReLU()
        self.conv_refine4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_refine4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine4_3 = nn.PReLU()
        self.conv_r4_1 = nn.Conv2d(512, 64, 3, padding=1)
        self.bn_r4_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_r4_1 = nn.PReLU()
        # DRB 5
        self.conv_refine5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_refine5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine5_1 = nn.PReLU()
        self.conv_refine5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_refine5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine5_2 = nn.PReLU()
        self.conv_refine5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_refine5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu_refine5_3 = nn.PReLU()
        self.conv_r5_1 = nn.Conv2d(512, 64, 3, padding=1)
        self.bn_r5_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_r5_1 = nn.PReLU()


        # -----------------------------  Multi-scale  ----------------------------- #
        # Add new structure: ASPP   Atrous spatial Pyramid Pooling     based on DeepLab v3
        # part0:   1*1*64 Conv
        self.conv5_conv_1 = nn.Conv2d(64, 64, 1, padding=0)              # size:  64*64*64
        self.bn5_conv_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_conv_1 = nn.ReLU(inplace=True)
        # part1:   3*3*64 Conv
        self.conv5_conv = nn.Conv2d(64, 64, 3, padding=1)                # size:  64*64*64
        self.bn5_conv = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_conv = nn.ReLU(inplace=True)
        # part2:   3*3*64 (dilated=7) Atrous Conv
        self.Atrous_conv_1 = nn.Conv2d(64, 64, 3, padding=7, dilation=7) # size:  64*64*64
        self.Atrous_bn5_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_1 = nn.ReLU(inplace=True)
        # part3:   3*3*64 (dilated=5) Atrous Conv
        self.Atrous_conv_2 = nn.Conv2d(64, 64, 3, padding=5, dilation=5) # size:  64*64*64
        self.Atrous_bn5_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_2 = nn.ReLU(inplace=True)
        # part4:   3*3*64 (dilated=3) Atrous Conv
        self.Atrous_conv_5 = nn.Conv2d(64, 64, 3, padding=3, dilation=3) # size:  64*64*64
        self.Atrous_bn5_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_5 = nn.ReLU(inplace=True)
        # part5:   Max_pooling                                           # size:  16*16*64
        self.Atrous_pooling = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.Atrous_conv_pool = nn.Conv2d(64, 64, 1, padding=0)
        self.Atrous_bn_pool = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_relu_pool = nn.ReLU(inplace=True)



        # -----------------------------  Channel-wise Attention  ----------------------------- #
        self.conv_c = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_h = nn.Conv2d(64, 64, 3, padding=1)
        self.pool_avg = nn.AvgPool2d(64, stride=2, ceil_mode=True)  # 1/8



        # -----------------------------  Sptatial-wise Attention  ----------------------------- #
        self.conv_s1 = nn.Conv2d(64 * self.num_layers, 64, 1, padding=0)
        self.conv_s2 = nn.Conv2d(64 * self.num_layers, 1, 1, padding=0)


        # -----------------------------  Prediction  ----------------------------- #
        self.conv_pred = nn.Conv2d(64, 2, 1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self,depth_vector,h1,h2,h3,h4,h5,d1,d2,d3,d4,d5):
        internal_state = []


        # -------- apply DRB --------- #
        # drb 1
        d1_1 = self.relu_refine1_1(self.bn_refine1_1(self.conv_refine1_1(d1)))
        d1_2 = self.relu_refine1_2(self.bn_refine1_2(self.conv_refine1_2(d1_1)))
        d1_2 = d1_2 + h1                                        # (256x256)*64
        d1_2 = self.down_2_2(self.down_2_1(d1_2))
        d1_2_0 = d1_2
        d1_3 = self.relu_refine1_3(self.bn_refine1_3(self.conv_refine1_3(d1_2)))
        drb1 = d1_2_0 + d1_3                                    # (64 x 64)*64

        # drb 2
        d2_1 = self.relu_refine2_1(self.bn_refine2_1(self.conv_refine2_1(d2)))
        d2_2 = self.relu_refine2_2(self.bn_refine2_2(self.conv_refine2_2(d2_1)))
        d2_2 = d2_2 + h2                                        # (128x128)*128
        d2_2 = self.down_2_1(d2_2)
        d2_2_0 = d2_2
        d2_3 = self.relu_refine2_3(self.bn_refine2_3(self.conv_refine2_3(d2_2)))
        drb2 = d2_2_0 + d2_3
        drb2 = self.relu_r2_1(self.bn_r2_1(self.conv_r2_1(drb2)))   # (64 x 64)*64

        # drb 3
        d3_1 = self.relu_refine3_1(self.bn_refine3_1(self.conv_refine3_1(d3)))
        d3_2 = self.relu_refine3_2(self.bn_refine3_2(self.conv_refine3_2(d3_1)))
        d3_2 = d3_2 + h3                                        # (64 x 64)*256
        d3_2_0 = d3_2
        d3_3 = self.relu_refine3_3(self.bn_refine3_3(self.conv_refine3_3(d3_2)))
        drb3 = d3_2_0 + d3_3
        drb3 = self.relu_r3_1(self.bn_r3_1(self.conv_r3_1(drb3)))   # (64 x 64)*64

        # drb 4
        d4_1 = self.relu_refine4_1(self.bn_refine4_1(self.conv_refine4_1(d4)))
        d4_2 = self.relu_refine4_2(self.bn_refine4_2(self.conv_refine4_2(d4_1)))
        d4_2 = d4_2 + h4                                        # (32 x 32)*512
        d4_2 =  F.upsample(d4_2, scale_factor=2, mode='bilinear')
        d4_2_0 = d4_2
        d4_3 = self.relu_refine4_3(self.bn_refine4_3(self.conv_refine4_3(d4_2)))
        drb4 = d4_2_0 + d4_3
        drb4 = self.relu_r4_1(self.bn_r4_1(self.conv_r4_1(drb4)))   # (64 x 64)*64

        # drb 5
        d5_1 = self.relu_refine5_1(self.bn_refine5_1(self.conv_refine5_1(d5)))
        d5_2 = self.relu_refine5_2(self.bn_refine5_2(self.conv_refine5_2(d5_1)))
        d5_2 = d5_2 + h5                                        # (16 x 16)*64
        d5_2 = F.upsample(d5_2, scale_factor=4, mode='bilinear')
        d5_2_0 = d5_2
        d5_3 = self.relu_refine5_3(self.bn_refine5_3(self.conv_refine5_3(d5_2)))
        drb5 = d5_2_0 + d5_3
        drb5 = self.relu_r5_1(self.bn_r5_1(self.conv_r5_1(drb5)))   # (64 x 64)*64

        drb_fusion = drb1 +drb2 + drb3 +drb4 +drb5              # (64 x 64)*64


        # --------------------- obtain multi-scale ----------------------- #
        f1 = self.relu5_conv_1(self.bn5_conv_1(self.conv5_conv_1(drb_fusion)))
        f2 = self.relu5_conv(self.bn5_conv(self.conv5_conv(drb_fusion)))
        f3 = self.Atrous_relu_1(self.Atrous_bn5_1(self.Atrous_conv_1(drb_fusion)))
        f4 = self.Atrous_relu_2(self.Atrous_bn5_2(self.Atrous_conv_2(drb_fusion)))
        f5 = self.Atrous_relu_5(self.Atrous_bn5_5(self.Atrous_conv_5(drb_fusion)))
        f6 = F.upsample(
            self.Atrous_relu_pool(self.Atrous_bn_pool(self.Atrous_conv_pool(self.Atrous_pooling(self.Atrous_pooling(drb_fusion))))),
            scale_factor=4, mode='bilinear')




        fusion = torch.cat([f1,f2,f3,f4,f5,f6],dim=0)  # 6x64x64x64
        fusion_o = fusion
        input = torch.cat(torch.chunk(fusion, 6, dim=0), dim=1)




        for step in range(self.step):
            depth = depth_vector                        # 1x 6 x 1 x1

            if step == 0:
                basize, _, height, width = input.size()
                (h_step, c) = ConvLSTMCell.init_hidden(basize, self.hidden_channels[self.num_layers-1],(height, width))


            # Feature-wise Attention
            depth = torch.mul(F.softmax(depth,dim=1), 6)

            basize, dime, h, w = depth.size()

            depth = depth.view(1, basize, dime, h, w).transpose(0,1).transpose(1,2)
            depth = torch.cat(torch.chunk(depth, basize, dim=0), dim=1).view(basize*dime, 1, 1, 1)

            depth = torch.mul(fusion_o, depth).view(1, basize*dime, 64, 64, 64)
            depth = torch.cat(torch.chunk(depth, basize, dim=1), dim=0)
            F_sum = torch.sum(depth, 1, keepdim=False)#.squeeze()


            # Channel-wise Attention
            depth_fw_ori = F_sum
            depth = self.conv_c(F_sum)
            h_c = self.conv_h(h_step)
            depth = depth + h_c
            depth = self.pool_avg(depth)
            depth = torch.mul(F.softmax(depth, dim=1), 64)
            F_sum_wt = torch.mul(depth_fw_ori, depth)



            x = F_sum_wt
            if step < self.step-1:
                for i in range(self.num_layers):
                    # all cells are initialized in the first step
                    if step == 0:
                        bsize, _, height, width = x.size()
                        (h, c) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[i], (height, width))
                        internal_state.append((h, c))
                    # do forward
                    name = 'cell{}'.format(i)
                    (h, c) = internal_state[i]
                    h_step = h

                    x, new_c, new_o = getattr(self, name)(x, h, c) # ConvLSTMCell forward
                    internal_state[i] = (x, new_c)

                # only record effective steps
                #if step in self.effective_step:

                if step == 0:
                    outputs_o = new_o
                else:
                    outputs_o = torch.cat((outputs_o, new_o), dim=1)

        # ---------------> Spatial-wise Attention Module <----------------- #
        outputs = self.conv_s1(outputs_o)
        spatial_weight = F.sigmoid(self.conv_s2(outputs_o))
        outputs = torch.mul(outputs,spatial_weight)
        # -------------------------> Prediction <-------------------------- #
        outputs = self.conv_pred(outputs)
        output = F.upsample(outputs, scale_factor=4, mode='bilinear')

        return output

