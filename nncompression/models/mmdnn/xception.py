import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.block1_conv1 = self.__conv(2, name='block1_conv1', in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.block1_conv1_bn = self.__batch_normalization(2, 'block1_conv1_bn', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.block1_conv2 = self.__conv(2, name='block1_conv2', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.block1_conv2_bn = self.__batch_normalization(2, 'block1_conv2_bn', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.block2_sepconv1_bn = self.__batch_normalization(2, 'block2_sepconv1_bn', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_1 = self.__batch_normalization(2, 'batch_normalization_1', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.block2_sepconv2_bn = self.__batch_normalization(2, 'block2_sepconv2_bn', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_2 = self.__batch_normalization(2, 'batch_normalization_2', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.block3_sepconv1_bn = self.__batch_normalization(2, 'block3_sepconv1_bn', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.block3_sepconv2_bn = self.__batch_normalization(2, 'block3_sepconv2_bn', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=256, out_channels=728, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_3 = self.__batch_normalization(2, 'batch_normalization_3', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block4_sepconv1_bn = self.__batch_normalization(2, 'block4_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block4_sepconv2_bn = self.__batch_normalization(2, 'block4_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block5_sepconv1_bn = self.__batch_normalization(2, 'block5_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block5_sepconv2_bn = self.__batch_normalization(2, 'block5_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block5_sepconv3_bn = self.__batch_normalization(2, 'block5_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block6_sepconv1_bn = self.__batch_normalization(2, 'block6_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block6_sepconv2_bn = self.__batch_normalization(2, 'block6_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block6_sepconv3_bn = self.__batch_normalization(2, 'block6_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block7_sepconv1_bn = self.__batch_normalization(2, 'block7_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block7_sepconv2_bn = self.__batch_normalization(2, 'block7_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block7_sepconv3_bn = self.__batch_normalization(2, 'block7_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block8_sepconv1_bn = self.__batch_normalization(2, 'block8_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block8_sepconv2_bn = self.__batch_normalization(2, 'block8_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block8_sepconv3_bn = self.__batch_normalization(2, 'block8_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block9_sepconv1_bn = self.__batch_normalization(2, 'block9_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block9_sepconv2_bn = self.__batch_normalization(2, 'block9_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block9_sepconv3_bn = self.__batch_normalization(2, 'block9_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block10_sepconv1_bn = self.__batch_normalization(2, 'block10_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block10_sepconv2_bn = self.__batch_normalization(2, 'block10_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block10_sepconv3_bn = self.__batch_normalization(2, 'block10_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block11_sepconv1_bn = self.__batch_normalization(2, 'block11_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block11_sepconv2_bn = self.__batch_normalization(2, 'block11_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block11_sepconv3_bn = self.__batch_normalization(2, 'block11_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block12_sepconv1_bn = self.__batch_normalization(2, 'block12_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block12_sepconv2_bn = self.__batch_normalization(2, 'block12_sepconv2_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block12_sepconv3_bn = self.__batch_normalization(2, 'block12_sepconv3_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=728, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.batch_normalization_4 = self.__batch_normalization(2, 'batch_normalization_4', num_features=1024, eps=0.0010000000474974513, momentum=0.0)
        self.block13_sepconv1_bn = self.__batch_normalization(2, 'block13_sepconv1_bn', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.block13_sepconv2_bn = self.__batch_normalization(2, 'block13_sepconv2_bn', num_features=1024, eps=0.0010000000474974513, momentum=0.0)
        self.block14_sepconv1_bn = self.__batch_normalization(2, 'block14_sepconv1_bn', num_features=1536, eps=0.0010000000474974513, momentum=0.0)
        self.block14_sepconv2_bn = self.__batch_normalization(2, 'block14_sepconv2_bn', num_features=2048, eps=0.0010000000474974513, momentum=0.0)
        self.predictions = self.__dense(name = 'predictions', in_features = 2048, out_features = 1000, bias = True)

    def forward(self, x):
        block1_conv1    = self.block1_conv1(x)
        block1_conv1_bn = self.block1_conv1_bn(block1_conv1)
        block1_conv1_act = F.relu(block1_conv1_bn)
        block1_conv2    = self.block1_conv2(block1_conv1_act)
        block1_conv2_bn = self.block1_conv2_bn(block1_conv2)
        block1_conv2_act = F.relu(block1_conv2_bn)
        conv2d_1        = self.conv2d_1(block1_conv2_act)
        block2_sepconv1_bn = self.block2_sepconv1_bn(block2_sepconv1)
        batch_normalization_1 = self.batch_normalization_1(conv2d_1)
        block2_sepconv2_act = F.relu(block2_sepconv1_bn)
        block2_sepconv2_bn = self.block2_sepconv2_bn(block2_sepconv2)
        block2_pool_pad = F.pad(block2_sepconv2_bn, (1, 1, 1, 1), value=float('-inf'))
        block2_pool, block2_pool_idx = F.max_pool2d(block2_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_1           = block2_pool + batch_normalization_1
        block3_sepconv1_act = F.relu(add_1)
        conv2d_2        = self.conv2d_2(add_1)
        batch_normalization_2 = self.batch_normalization_2(conv2d_2)
        block3_sepconv1_bn = self.block3_sepconv1_bn(block3_sepconv1)
        block3_sepconv2_act = F.relu(block3_sepconv1_bn)
        block3_sepconv2_bn = self.block3_sepconv2_bn(block3_sepconv2)
        block3_pool_pad = F.pad(block3_sepconv2_bn, (0, 1, 0, 1), value=float('-inf'))
        block3_pool, block3_pool_idx = F.max_pool2d(block3_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_2           = block3_pool + batch_normalization_2
        block4_sepconv1_act = F.relu(add_2)
        conv2d_3        = self.conv2d_3(add_2)
        batch_normalization_3 = self.batch_normalization_3(conv2d_3)
        block4_sepconv1_bn = self.block4_sepconv1_bn(block4_sepconv1)
        block4_sepconv2_act = F.relu(block4_sepconv1_bn)
        block4_sepconv2_bn = self.block4_sepconv2_bn(block4_sepconv2)
        block4_pool_pad = F.pad(block4_sepconv2_bn, (1, 1, 1, 1), value=float('-inf'))
        block4_pool, block4_pool_idx = F.max_pool2d(block4_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_3           = block4_pool + batch_normalization_3
        block5_sepconv1_act = F.relu(add_3)
        block5_sepconv1_bn = self.block5_sepconv1_bn(block5_sepconv1)
        block5_sepconv2_act = F.relu(block5_sepconv1_bn)
        block5_sepconv2_bn = self.block5_sepconv2_bn(block5_sepconv2)
        block5_sepconv3_act = F.relu(block5_sepconv2_bn)
        block5_sepconv3_bn = self.block5_sepconv3_bn(block5_sepconv3)
        add_4           = block5_sepconv3_bn + add_3
        block6_sepconv1_act = F.relu(add_4)
        block6_sepconv1_bn = self.block6_sepconv1_bn(block6_sepconv1)
        block6_sepconv2_act = F.relu(block6_sepconv1_bn)
        block6_sepconv2_bn = self.block6_sepconv2_bn(block6_sepconv2)
        block6_sepconv3_act = F.relu(block6_sepconv2_bn)
        block6_sepconv3_bn = self.block6_sepconv3_bn(block6_sepconv3)
        add_5           = block6_sepconv3_bn + add_4
        block7_sepconv1_act = F.relu(add_5)
        block7_sepconv1_bn = self.block7_sepconv1_bn(block7_sepconv1)
        block7_sepconv2_act = F.relu(block7_sepconv1_bn)
        block7_sepconv2_bn = self.block7_sepconv2_bn(block7_sepconv2)
        block7_sepconv3_act = F.relu(block7_sepconv2_bn)
        block7_sepconv3_bn = self.block7_sepconv3_bn(block7_sepconv3)
        add_6           = block7_sepconv3_bn + add_5
        block8_sepconv1_act = F.relu(add_6)
        block8_sepconv1_bn = self.block8_sepconv1_bn(block8_sepconv1)
        block8_sepconv2_act = F.relu(block8_sepconv1_bn)
        block8_sepconv2_bn = self.block8_sepconv2_bn(block8_sepconv2)
        block8_sepconv3_act = F.relu(block8_sepconv2_bn)
        block8_sepconv3_bn = self.block8_sepconv3_bn(block8_sepconv3)
        add_7           = block8_sepconv3_bn + add_6
        block9_sepconv1_act = F.relu(add_7)
        block9_sepconv1_bn = self.block9_sepconv1_bn(block9_sepconv1)
        block9_sepconv2_act = F.relu(block9_sepconv1_bn)
        block9_sepconv2_bn = self.block9_sepconv2_bn(block9_sepconv2)
        block9_sepconv3_act = F.relu(block9_sepconv2_bn)
        block9_sepconv3_bn = self.block9_sepconv3_bn(block9_sepconv3)
        add_8           = block9_sepconv3_bn + add_7
        block10_sepconv1_act = F.relu(add_8)
        block10_sepconv1_bn = self.block10_sepconv1_bn(block10_sepconv1)
        block10_sepconv2_act = F.relu(block10_sepconv1_bn)
        block10_sepconv2_bn = self.block10_sepconv2_bn(block10_sepconv2)
        block10_sepconv3_act = F.relu(block10_sepconv2_bn)
        block10_sepconv3_bn = self.block10_sepconv3_bn(block10_sepconv3)
        add_9           = block10_sepconv3_bn + add_8
        block11_sepconv1_act = F.relu(add_9)
        block11_sepconv1_bn = self.block11_sepconv1_bn(block11_sepconv1)
        block11_sepconv2_act = F.relu(block11_sepconv1_bn)
        block11_sepconv2_bn = self.block11_sepconv2_bn(block11_sepconv2)
        block11_sepconv3_act = F.relu(block11_sepconv2_bn)
        block11_sepconv3_bn = self.block11_sepconv3_bn(block11_sepconv3)
        add_10          = block11_sepconv3_bn + add_9
        block12_sepconv1_act = F.relu(add_10)
        block12_sepconv1_bn = self.block12_sepconv1_bn(block12_sepconv1)
        block12_sepconv2_act = F.relu(block12_sepconv1_bn)
        block12_sepconv2_bn = self.block12_sepconv2_bn(block12_sepconv2)
        block12_sepconv3_act = F.relu(block12_sepconv2_bn)
        block12_sepconv3_bn = self.block12_sepconv3_bn(block12_sepconv3)
        add_11          = block12_sepconv3_bn + add_10
        block13_sepconv1_act = F.relu(add_11)
        conv2d_4        = self.conv2d_4(add_11)
        batch_normalization_4 = self.batch_normalization_4(conv2d_4)
        block13_sepconv1_bn = self.block13_sepconv1_bn(block13_sepconv1)
        block13_sepconv2_act = F.relu(block13_sepconv1_bn)
        block13_sepconv2_bn = self.block13_sepconv2_bn(block13_sepconv2)
        block13_pool_pad = F.pad(block13_sepconv2_bn, (1, 1, 1, 1), value=float('-inf'))
        block13_pool, block13_pool_idx = F.max_pool2d(block13_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_12          = block13_pool + batch_normalization_4
        block14_sepconv1_bn = self.block14_sepconv1_bn(block14_sepconv1)
        block14_sepconv1_act = F.relu(block14_sepconv1_bn)
        block14_sepconv2_bn = self.block14_sepconv2_bn(block14_sepconv2)
        block14_sepconv2_act = F.relu(block14_sepconv2_bn)
        avg_pool        = F.avg_pool2d(input = block14_sepconv2_act, kernel_size = block14_sepconv2_act.size()[2:])
        avg_pool_flatten = avg_pool.view(avg_pool.size(0), -1)
        predictions     = self.predictions(avg_pool_flatten)
        predictions_activation = F.softmax(predictions)
        return predictions_activation


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

