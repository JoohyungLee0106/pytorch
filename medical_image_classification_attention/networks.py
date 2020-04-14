import torch
import torch.nn as nn
import math
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# https://pytorch.org/hub/pytorch_vision_resnet/
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://pytorch.org/docs/stable/nn.init.html
# https://pytorch.org/docs/stable/nn.html#conv2d
# https://pytorch.org/docs/stable/nn.functional.html
# https://github.com/fyu/drn/blob/master/drn.py
#


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

#################################################################################################


class ResNet_2D_attention(nn.Module):
    def __init__(self, arch, block, layers, dilations=False, lated=False, groups=1, base_width=64, global_pooling='avp',
                 starting_filter_num=64, pretrained=True, attention_threshold=0.5, strict=False,
                 progress=False, my_device=torch.device("cuda:0"), if_single_input_channel=True, **kwargs):
        '''

        :param arch:
        :param block:
        :param layers:
        :param dilations:
        :param groups:
        :param base_width:
        :param global_pooling:
        :param starting_filter_num:
        :param pretrained:
        :param attention_threshold:
        :param attention_size: [C, H], for H=W
        :param strict:
        :param progress:
        '''
        super(ResNet_2D_attention, self).__init__()
        num_classes = 1
        self.my_device = my_device
        self.starting_filter_num = starting_filter_num
        self.model_resnet = ResNet(block, layers, groups=groups, starting_filter_num=starting_filter_num,
                                   width_per_group=base_width, if_single_input_channel=if_single_input_channel, **kwargs)
        self.attention_threshold = attention_threshold
        self.dilations = dilations

        if lated:
            if dilations:
                self.forward = self.forward_dilation_lated
            else:
                self.forward = self.forward_without_dilation_lated
            self.conv_attention = nn.Conv2d(starting_filter_num * 4 * block.expansion, 1, kernel_size=1, bias=False)
        else:
            if dilations:
                self.forward = self.forward_dilation
            else:
                self.forward = self.forward_without_dilation
            self.conv_attention = nn.Conv2d(starting_filter_num * 2 * block.expansion, 1, kernel_size=1, bias=False)

        if pretrained:
            if starting_filter_num is not 64:
                raise ValueError("starting_filter_num must be 64 for pre-training")
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            if block == Bottleneck:
                state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
            if if_single_input_channel:
                state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)

            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
            state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
            state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")

            self.model_resnet.load_state_dict(state_dict, strict=True)

        if (global_pooling == 'mxp'):
            self.globalpool = nn.AdaptiveMaxPool2d((1, 1))
        elif (global_pooling == 'avp'):
            self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        elif (global_pooling == 'partial_avp'):
            self.globalpool = Adaptive_partial_AvgPool2d()
        else:
            raise ValueError("<Class ResNet> Invalid Argument: global_pooling")

        # self.conv_cancer = nn.Conv2d(starting_filter_num * 4 * block.expansion, 1, kernel_size=1, bias=False)
        self.fc = nn.Linear(starting_filter_num * 8 * block.expansion, num_classes)
        self.sigmoid_attention = nn.Sigmoid()
        # Do I need separate nn.Sigmoid instances?
        # self.sigmoid_cancer = nn.Sigmoid()
        # self.sigmoid_classification = nn.Sigmoid()

    def get_attention(self, x, layer, stride, dilation):
        x = layer(x, stride=stride, dilation=dilation)
        attention_logits = self.conv_attention(x)
        attention_prob = self.sigmoid_attention(attention_logits)
        # Hard Attention
        # attention_mask = F.relu(attention_logits).sign()
        # attention_mask = attention_mask.repeat(1, x.size(1), 1, 1)

        # Soft Attention
        attention_mask = torch.where(attention_prob > self.attention_threshold, attention_prob,
                                      torch.zeros(attention_prob.size()).to(self.my_device))

        attention_mask = attention_mask.repeat(1, x.size(1), 1, 1)
        return x, attention_logits, attention_mask

    def get_classification(self, x):
        # layer#(input, stride, dilation)
        out = self.model_resnet.layer4(x, 2, [1, 1])
        out = self.globalpool(out)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        # out = self.sigmoid_classification(out)
        out = out.view(out.size(0))
        return out

    def forward_without_dilation(self, x):
        x = self.model_resnet.forward_shared(x)
        x, attention_logits, attention_mask = self.get_attention(x, self.model_resnet.layer2, 2, [1, 1])
        x = self.model_resnet.layer3(torch.mul(x, attention_mask), 2, [1, 1])
        clasf_logits = self.get_classification(x)
        attention_logits = torch.squeeze(attention_logits, dim=1)
        return clasf_logits, attention_logits

    def forward_without_dilation_lated(self, x):
        x = self.model_resnet.forward_shared(x)
        x = self.model_resnet.layer2(x, 2, [1, 1])
        x, attention_logits, attention_mask = self.get_attention(x, self.model_resnet.layer3, 2, [1, 1])
        clasf_logits = self.get_classification(torch.mul(x, attention_mask))
        attention_logits = torch.squeeze(attention_logits, dim=1)
        return clasf_logits, attention_logits

    def forward_dilation(self, x):
        x = self.model_resnet.forward_shared(x)
        x_seg, attention_logits, attention_mask = self.get_attention(x, self.model_resnet.layer2, 1, [1, 2])
        # linear vs. bilinear vs. bicubic
        # Nearest available iff align_corners=False
        attention_mask = F.interpolate(attention_mask, int(x_seg.size(3) / 2), mode='bicubic', align_corners=True)

        x = self.model_resnet.layer2(x, 2, [1, 1])
        x = self.model_resnet.layer3(torch.mul(x, attention_mask), 2, [1, 1])
        clasf_logits = self.get_classification(x)
        attention_logits = torch.squeeze(attention_logits, dim=1)
        return clasf_logits, attention_logits

    def forward_dilation_lated(self, x):
        x = self.model_resnet.forward_shared(x)
        x_seg = self.model_resnet.layer2(x, 1, [1, 2])
        x_seg, attention_logits, attention_mask = self.get_attention(x_seg, self.model_resnet.layer3, 1, [2, 4])
        # linear vs. bilinear vs. bicubic
        # Nearest available iff align_corners=False
        attention_mask = F.interpolate(attention_mask, int(x_seg.size(3) / 4), mode='bicubic', align_corners=True)

        x = self.model_resnet.layer2(x, 2, [1, 1])
        x = self.model_resnet.layer3(x, 2, [1, 1])
        clasf_logits = self.get_classification(torch.mul(x, attention_mask))
        attention_logits = torch.squeeze(attention_logits, dim=1)
        return clasf_logits, attention_logits


class ResNet_2D_attention_with_cancer(ResNet_2D_attention):
    def __init__(self, arch, block, layers, dilations=False, lated=False, groups=1, base_width=64, global_pooling='avp',
                 starting_filter_num=64, pretrained=True, attention_threshold=0.5, strict=False,
                 progress=False, my_device=torch.device("cuda:0"), if_single_input_channel=True, **kwargs):

        super(ResNet_2D_attention_with_cancer, self).__init__(arch, block, layers, dilations, lated, groups,
                                                              base_width, global_pooling, starting_filter_num,
                                                              pretrained, attention_threshold, strict, progress,
                                                              my_device, if_single_input_channel, **kwargs)

        if lated:
            self.conv_cancer = nn.Conv2d(self.starting_filter_num * 8 * block.expansion, 1, kernel_size=1, bias=False)
        else:
            self.conv_cancer = nn.Conv2d(self.starting_filter_num * 4 * block.expansion, 1, kernel_size=1, bias=False)

        # Do I need separate nn.Sigmoid instances?
        self.sigmoid_cancer = nn.Sigmoid()

    def get_attention(self, x, layer, stride, dilation):
        x = layer(x, stride=stride, dilation=dilation)
        attention_prob = self.sigmoid_attention(self.conv_attention(x))

        # Hard Attention
        # attention_mask = F.relu(attention_logits).sign()
        # attention_mask = attention_mask.repeat(1, x.size(1), 1, 1)

        # Soft Attention
        # attention_mask = torch.where(attention_prob > self.attention_threshold, attention_prob,
        #                               torch.zeros(attention_prob.size()).to(self.my_device))

        return x, attention_prob

    def get_classification(self, x):

        # layer#(input, stride, dilation)
        out = self.globalpool(x)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        out = out.view(out.size(0))
        # out = self.sigmoid_classification(out)

        return out

    def forward_simultaneous_without_dilation_lated(self, x):
        x = self.model_resnet.forward_shared(x)
        x = self.model_resnet.layer2(x, 2, [1, 1])
        x, attention_prob = self.get_attention(x, self.model_resnet.layer3, 2, [1, 1])
        attention_mask_clasf = torch.where(attention_prob > self.attention_threshold, attention_prob,
                                           torch.zeros(attention_prob.size()).to(self.my_device))
        attention_mask_clasf = attention_mask_clasf.repeat(1, x.size(1), 1, 1)

        x = self.model_resnet.layer4(torch.mul(x, attention_mask_clasf), 2, [1, 1])
        cancer_prob = self.sigmoid_cancer(self.conv_cancer(x))
        attention_mask_seg = F.interpolate(attention_prob, cancer_prob.size(3), mode='bicubic', align_corners=True)
        attention_mask_seg = torch.where(attention_mask_seg > self.attention_threshold, attention_mask_seg,
                                         torch.zeros(attention_mask_seg.size()).to(self.my_device))
        cancer_prob = torch.mul(cancer_prob, attention_mask_seg)
        self.attention_mask_seg = attention_mask_seg

        clasf_logits = self.get_classification(x)
        attention_prob = torch.squeeze(attention_prob, dim=1)
        cancer_prob = torch.squeeze(cancer_prob, dim=1)
        return clasf_logits, attention_prob, cancer_prob

    # def forward_simultaneous_with_dilation_lated(self, x):
    #
    # def forward_without_dilation(self, x):
    #     x = self.model_resnet.forward_shared(x)
    #     x, attention_prob = self.get_attention(x, self.model_resnet.layer2, 2, [1, 1])
    #     attention_mask_clasf = torch.where(attention_prob > self.attention_threshold, attention_prob,
    #                                  torch.zeros(attention_prob.size()).to(self.my_device))
    #     attention_mask_clasf = attention_mask_clasf.repeat(1, x.size(1), 1, 1)
    #
    #     x = self.model_resnet.layer3(torch.mul(x, attention_mask_clasf), 2, [1, 1])
    #     cancer_prob = self.sigmoid_cancer(self.conv_cancer(x))
    #     attention_mask_seg = F.interpolate(attention_prob, cancer_prob.size(3), mode='bicubic', align_corners=True)
    #     attention_mask_seg = torch.where(attention_mask_seg > self.attention_threshold, attention_mask_seg,
    #                                  torch.zeros(attention_mask_seg.size()).to(self.my_device))
    #     cancer_prob = torch.mul(cancer_prob, attention_mask_seg)
    #     self.attention_mask_seg = attention_mask_seg
    #
    #     x = self.model_resnet.layer4(x, 2, [1, 1])
    #     clasf_logits = self.get_classification(x)
    #     attention_prob = torch.squeeze(attention_prob, dim=1)
    #     cancer_prob = torch.squeeze(cancer_prob, dim=1)
    #
        # return clasf_logits, attention_prob, cancer_prob

    def forward_without_dilation_lated(self, x):
        x = self.model_resnet.forward_shared(x)
        x = self.model_resnet.layer2(x, 2, [1, 1])
        x, attention_prob = self.get_attention(x, self.model_resnet.layer3, 2, [1, 1])
        attention_mask_clasf = torch.where(attention_prob > self.attention_threshold, attention_prob,
                                           torch.zeros(attention_prob.size()).to(self.my_device))
        attention_mask_clasf = attention_mask_clasf.repeat(1, x.size(1), 1, 1)

        x = self.model_resnet.layer4(torch.mul(x, attention_mask_clasf), 2, [1, 1])
        cancer_prob = self.sigmoid_cancer(self.conv_cancer(x))
        attention_mask_seg = F.interpolate(attention_prob, cancer_prob.size(3), mode='bicubic', align_corners=True)
        attention_mask_seg = torch.where(attention_mask_seg > self.attention_threshold, attention_mask_seg,
                                     torch.zeros(attention_mask_seg.size()).to(self.my_device))
        cancer_prob = torch.mul(cancer_prob, attention_mask_seg)
        self.attention_mask_seg = attention_mask_seg

        clasf_logits = self.get_classification(x)
        attention_prob = torch.squeeze(attention_prob, dim=1)
        cancer_prob = torch.squeeze(cancer_prob, dim=1)
        return clasf_logits, attention_prob, cancer_prob

    def forward_dilation(self, x):
        x = self.model_resnet.forward_shared(x)
        x_seg, attention_prob = self.get_attention(x, self.model_resnet.layer2, 1, [1, 2])

        x_seg = self.model_resnet.layer3(x_seg, 1, [2, 4])
        cancer_prob = self.sigmoid_cancer(self.conv_cancer(x_seg))
        attention_mask_seg = torch.where(attention_prob > self.attention_threshold, attention_prob,
                                     torch.zeros(attention_prob.size()).to(self.my_device))
        cancer_prob = torch.mul(cancer_prob, attention_mask_seg)
        self.attention_mask_seg = attention_mask_seg

        x = self.model_resnet.layer2(x, 2, [1, 1])
        attention_mask_clasf = F.interpolate(attention_prob, x.size(3), mode='bicubic', align_corners=True)
        attention_mask_clasf = torch.where(attention_mask_clasf > self.attention_threshold, attention_mask_clasf,
                                           torch.zeros(attention_mask_clasf.size()).to(self.my_device))
        attention_mask_clasf = attention_mask_clasf.repeat(1, x.size(1), 1, 1)

        x = self.model_resnet.layer3(torch.mul(x, attention_mask_clasf), 2, [1, 1])
        x = self.model_resnet.layer4(x, 2, [1, 1])
        clasf_logits = self.get_classification(x)
        attention_prob = torch.squeeze(attention_prob, dim=1)
        cancer_prob = torch.squeeze(cancer_prob, dim=1)

        return clasf_logits, attention_prob, cancer_prob

    def forward_dilation_lated(self, x):
        x = self.model_resnet.forward_shared(x)
        x_seg = self.model_resnet.layer2(x, 1, [1, 2])
        x_seg, attention_prob = self.get_attention(x_seg, self.model_resnet.layer3, 1, [2, 4])
        x_seg = self.model_resnet.layer4(x_seg, 1, [4, 8])
        cancer_prob = self.sigmoid_cancer(self.conv_cancer(x_seg))
        attention_mask_seg = torch.where(attention_prob > self.attention_threshold, attention_prob,
                                         torch.zeros(attention_prob.size()).to(self.my_device))
        cancer_prob = torch.mul(cancer_prob, attention_mask_seg)
        self.attention_mask_seg = attention_mask_seg

        x = self.model_resnet.layer2(x, 2, [1, 1])
        x = self.model_resnet.layer3(x, 2, [1, 1])
        attention_mask_clasf = F.interpolate(attention_prob, x.size(3), mode='bicubic', align_corners=True)
        attention_mask_clasf = torch.where(attention_mask_clasf > self.attention_threshold, attention_mask_clasf,
                                           torch.zeros(attention_mask_clasf.size()).to(self.my_device))
        attention_mask_clasf = attention_mask_clasf.repeat(1, x.size(1), 1, 1)

        x = self.model_resnet.layer4(torch.mul(x, attention_mask_clasf), 2, [1, 1])
        clasf_logits = self.get_classification(x)
        attention_prob = torch.squeeze(attention_prob, dim=1)
        cancer_prob = torch.squeeze(cancer_prob, dim=1)
        return clasf_logits, attention_prob, cancer_prob



#################################################################################################


class Sequential_downsample(nn.Sequential):
    def forward(self, input, stride=1):
        input = self[0](input, stride=stride)

        for module in self[1:]:
            input = module(input)
        return input

class Custom_sequential(nn.Sequential):
    # def __init__(self, *args):
    #     super(Custom_sequential, self).__init__()

    def forward(self, input, stride=2, dilation=[1,1]):
    # def forward(self, input, stride, dilation):
        # print("+++++++++++++")
        # print(self[0])
        # print("~~~~")
        # print(self[-1])
        # print(f'input size: {input.size()}, stride: {stride}, dilation: {dilation}')
        # print(f'stride: {stride}')
        input = self[0](input, stride=stride, dilation=[dilation[0], dilation[1]])

        for module in self[1:]:
            input = module(input, 1, [dilation[1], dilation[1]])
        return input

class Conv_downsample(nn.Module):
    def __init__(self, inplane, outplane, stride=1, groups=1):
        super(Conv_downsample, self).__init__()
        self.conv_downsample = nn.Conv2d(inplane, outplane, kernel_size=1, stride=stride, bias=False)
        self.groups=1
    def forward(self, x, stride):
        return F.conv2d(x, self.conv_downsample.weight, stride=stride, groups=self.groups)

class Adaptive_partial_AvgPool(nn.Module):
    '''
    Input must consist of non-negative numbers
    '''
    def __init__(self, threshold=0, image_dim=2):
        super(Adaptive_partial_AvgPool3d, self).__init__()
        self.threshold = threshold
        self.image_dim = image_dim
    def forward(self, x, keepdims=True):
        xx=x.clone()
        xx[xx>threshold]=1
        xx=torch.sum(xx, list(range(2, self.image_dim+2)), keepdim=keepdims)
        x=torch.sum(x, list(range(2, self.image_dim+2)), keepdim=keepdims)
        return torch.div(x, xx)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, starting_filter_num=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        outplanes = int(planes * base_width/64.0)
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # print(f'inplanes: {inplanes}, outplanes: {outplanes}')
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, stride, dilation):
        '''

        :param x: Input (NCHW)
        :param stride: Scalar
        :param dilation: Length-2 list
        :return:
        '''
        identity = x
        # print(f'input: {x.size()}')
        out = F.conv2d(x, self.conv1.weight, stride = stride, padding=dilation[0], dilation = dilation[0])
        out = self.bn1(out)
        out = self.relu(out)
        # print(f'conv1: {out.size()}')
        out = F.conv2d(out, self.conv2.weight, padding=dilation[1], dilation = dilation[1])
        out = self.bn2(out)
        # print(f'conv2: {out.size()}')
        if x.size() != out.size():
            # print(f'x size: {x.size()}, out size: {out.size()}, downsample: {self.downsample[0].conv_downsample.weight.size()}, stride: {int(x.size(3)/out.size(3))}')
            identity = self.downsample(x, stride=int(x.size(3)/out.size(3)))

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, starting_filter_num=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.) * (self.starting_filter_num / 64.)) * groups
        width = int(base_width * groups * float(planes)/64.0)
        outplane = int(starting_filter_num * planes * self.expansion /64.0)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        print(f'inplanes: {inplanes}, width: {width}, outplane: {outplane}')
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, outplane)
        self.bn3 = norm_layer(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.groups = groups


    def forward(self, x, stride, dilation):
        '''
        :param x: Input (NCHW)
        :param stride: Scalar
        :param dilation: Length-2 list
        :return:
        '''
        identity = x
        # print("In: "+str(x.size()))
        # print(f'conv1.weight size: {self.conv1.weight.size()}')
        out = F.conv2d(x, self.conv1.weight, dilation = dilation[0])
        out = self.bn1(out)
        out = self.relu(out)
        out = F.conv2d(out, self.conv2.weight, stride = stride, padding=dilation[1], dilation = dilation[1], groups=self.groups)
        out = self.bn2(out)
        out = self.relu(out)

        # default stride = 1
        out = F.conv2d(out, self.conv3.weight, dilation = dilation[1])

        out = self.bn3(out)

        if x.size() != out.size():
            # print(f"x.size(): {x.size()}")
            # print(f"out.size(): {out.size()}")
            # print(f'downsample conv weight size: {self.downsample[0].conv_downsample.weight.size()}')
            identity = self.downsample(x, stride=int(x.size(3)/out.size(3)))
        # print("Out: "+str(out.size()))
        # print(f'stride: {stride}, dilation[0]: {dilation[0]}, dilation[1]: {dilation[1]}')
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, starting_filter_num=64, width_per_group=64, global_pooling='avp',
                 norm_layer=None, if_fc=False, num_classes=1, if_single_input_channel=True):
        '''
        :param block:
        :param layers:
        :param num_classes:
        :param zero_init_residual:
        :param groups:
        :param starting_filter_num: Starting # of filters
        :param width_per_group:
        :param replace_stride_with_dilation:
        :param norm_layer: Default is nn.BatchNorm2d.
        Other norms can be used especially when the # of mini-batch is small
        :param global_pooling:
        '''
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = starting_filter_num
        self.starting_filter_num = starting_filter_num
        if if_single_input_channel:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if if_fc:
            self.fc = nn.Linear(starting_filter_num * 8 * block.expansion, num_classes)
            self.sigmoid = nn.Sigmoid()

            if (global_pooling == 'mxp'):
                self.globalpool = nn.AdaptiveMaxPool2d((1, 1))
            elif (global_pooling == 'avp'):
                self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
            elif (global_pooling == 'partial_avp'):
                self.globalpool = Adaptive_partial_AvgPool2d()
            else:
                raise ValueError("<Class ResNet> Invalid Argument: global_pooling")

    def _make_layer(self, block, planes, blocks, stride=1, sequential=Custom_sequential):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # if stride != 1 or self.inplanes != planes * block.expansion:
        if stride != 1 or self.inplanes != int(planes*self.starting_filter_num/64.0) * block.expansion:
            downsample = Sequential_downsample(
                Conv_downsample(self.inplanes, int(planes*self.starting_filter_num/64.0) * block.expansion),
                norm_layer(int(planes*self.starting_filter_num/64.0) * block.expansion),
            )
            print(f'inplane: {self.inplanes}, plane: {planes}, outplane: {int(planes*self.starting_filter_num/64.0) * block.expansion}')
        else:
            print(f'<else> inplane: {self.inplanes}, plane: {planes}, outplane: {int(planes*self.starting_filter_num/64.0) * block.expansion}')

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.starting_filter_num, previous_dilation, norm_layer))
        self.inplanes = int(self.starting_filter_num*planes*block.expansion/64.0)

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, starting_filter_num=self.starting_filter_num, dilation=self.dilation,
                                norm_layer=norm_layer))

        return sequential(*layers)


    def forward_shared(self, x):
        # See note [TorchScript super()]
        # print(f'<layer-wise> INPUT: {x.size()}')
        x = self.conv1(x)
        # print(f'<layer-wise> conv1: {x.size()}')
        x = self.bn1(x)
        # print(f'<layer-wise> bn1: {x.size()}')
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f'<layer-wise> maxpool: {x.size()}')
        return self.layer1(x, stride=1, dilation=[1, 1])

    def forward_separate(self, x):
        # print(f'<layer-wise> layer1: {x.size()}')
        x = self.layer2(x, stride=2, dilation=[1,1])
        # print(f'<layer-wise> layer2: {layer2_out.size()}')
        x = self.layer3(x, stride=2, dilation=[1,1])
        # print(f'<layer-wise> layer3: {layer3_out.size()}')
        x = self.layer4(x, stride=2, dilation=[1,1])
        # print(f'<layer-wise> layer4: {layer4_out.size()}')

        # Only use convs as a feature extractor
        # x = self.globalpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        x = self.forward_separate(self.forward_shared(x))
        x = self.globalpool(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        x = x.view(x.size(0))
        # x = self.sigmoid(x)

        return x

###############################################################################################

def resnet18(pretrained=False, starting_filter_num=64, zero_init_residual=False, if_single_input_channel=True, **kwargs):
    '''
    pretrained, starting_filter_num, zero_init_residual=False, if_single_input_channel 만 조정해주시면 됩니다!
    :param pretrained:
    :param starting_filter_num:
    :param zero_init_residual:
    :param if_single_input_channel:
    :param kwargs:
    :return:
    '''

    model = ResNet(BasicBlock, [2, 2, 2, 2], zero_init_residual=zero_init_residual,
                   starting_filter_num=starting_filter_num, width_per_group=starting_filter_num, if_fc=True,
                   if_single_input_channel = if_single_input_channel, **kwargs)
    arch = 'resnet18'
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=False)
        # print(f'layer1 downsample size: {state_dict["layer1.0.downsample.0.weight"].size()}')
        if if_single_input_channel:
            state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)
        # state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")
        print(f'before load: {model.conv1.weight.mean()}')
        model.load_state_dict(state_dict, strict=False)
        print(f'after load: {model.conv1.weight.mean()}')

    return model

def resnet34(pretrained=False, starting_filter_num=64, zero_init_residual=False, if_single_input_channel=True, **kwargs):
    '''
    pretrained, starting_filter_num, zero_init_residual=False, if_single_input_channel 만 조정해주시면 됩니다!
    :param pretrained:
    :param starting_filter_num:
    :param zero_init_residual:
    :param if_single_input_channel:
    :param kwargs:
    :return:
    '''
    model = ResNet(BasicBlock, [3,4,6,3], zero_init_residual=zero_init_residual,
                   starting_filter_num=starting_filter_num, width_per_group=starting_filter_num, if_fc=True,
                   if_single_input_channel = if_single_input_channel, **kwargs)
    arch = 'resnet34'
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=False)
        if if_single_input_channel:
            state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)
        # state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")

        model.load_state_dict(state_dict, strict=False)

    return model

def resnet50(pretrained=False, starting_filter_num=64, zero_init_residual=False, if_single_input_channel=True, **kwargs):
    '''
    pretrained, starting_filter_num, zero_init_residual=False, if_single_input_channel 만 조정해주시면 됩니다!
    :param pretrained:
    :param starting_filter_num:
    :param zero_init_residual:
    :param if_single_input_channel:
    :param kwargs:
    :return:
    '''
    model = ResNet(Bottleneck, [3,4,6,3], zero_init_residual=zero_init_residual,
                   starting_filter_num=starting_filter_num, width_per_group=starting_filter_num, if_fc=True,
                   if_single_input_channel = if_single_input_channel, **kwargs)
    arch = 'resnet50'
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=False)
        if if_single_input_channel:
            state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")
        model.load_state_dict(state_dict, strict=False)

    return model

def resnext50(pretrained=False, starting_filter_num=64, zero_init_residual=False, if_single_input_channel=True, **kwargs):
    '''
    pretrained, starting_filter_num, zero_init_residual=False, if_single_input_channel 만 조정해주시면 됩니다!
    :param pretrained:
    :param starting_filter_num:
    :param zero_init_residual:
    :param if_single_input_channel:
    :param kwargs:
    :return:
    '''
    if starting_filter_num < 16:
        raise ValueError("starting_filter_num  must be at least 16")
    ratio = float(starting_filter_num/64.0)
    model = ResNet(Bottleneck, [3,4,6,3], zero_init_residual=zero_init_residual, groups=int(32*ratio),
                   starting_filter_num=starting_filter_num, width_per_group=int(4*ratio), if_fc=True,
                   if_single_input_channel = if_single_input_channel, **kwargs)
    arch = 'resnext50_32x4d'
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=False)
        if if_single_input_channel:
            state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")

        model.load_state_dict(state_dict, strict=False)

    return model

def attention_dilated_2D(pretrained, dilations, global_pooling, arch='resnet34', block=BasicBlock, layers=[3,4,6,3], lated=False,
                         groups=1, base_width=64, starting_filter_num=64, attention_threshold=0.5, strict=False,
                 progress=False, my_device=torch.device("cuda:0"), if_single_input_channel=True, **kwargs):
    '''
    pretrained, starting_filter_num, zero_init_residual=False, if_single_input_channel 만 조정해주시면 됩니다!
    :param pretrained:
    :param starting_filter_num:
    :param zero_init_residual:
    :param if_single_input_channel:
    :param kwargs:
    :return:
    '''
    model = ResNet_2D_attention(arch=arch, block=block, layers=layers, dilations=dilations, lated=lated, groups=groups,
                                base_width=base_width, global_pooling=global_pooling,starting_filter_num=starting_filter_num,
                 pretrained=pretrained, attention_threshold=attention_threshold, strict=strict,
                 progress=progress, my_device=my_device, if_single_input_channel=if_single_input_channel, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=False)
        if if_single_input_channel:
            state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)
        # state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")

        model.load_state_dict(state_dict, strict=False)

    return model

def attention_dilated_with_cancer_2D(pretrained, dilations, global_pooling, arch='resnet34', block=BasicBlock, layers=[3,4,6,3], lated=False,
                         groups=1, base_width=64, starting_filter_num=64, attention_threshold=0.5, strict=False,
                 progress=False, my_device=torch.device("cuda:0"), if_single_input_channel=True, **kwargs):
    '''
    pretrained, starting_filter_num, zero_init_residual=False, if_single_input_channel 만 조정해주시면 됩니다!
    :param pretrained:
    :param starting_filter_num:
    :param zero_init_residual:
    :param if_single_input_channel:
    :param kwargs:
    :return:
    '''
    model = ResNet_2D_attention_with_cancer(arch=arch, block=block, layers=layers, dilations=dilations, lated=lated, groups=groups,
                                base_width=base_width, global_pooling=global_pooling,starting_filter_num=starting_filter_num,
                 pretrained=pretrained, attention_threshold=attention_threshold, strict=strict,
                 progress=progress, my_device=my_device, if_single_input_channel=if_single_input_channel, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=False)
        if if_single_input_channel:
            state_dict['conv1.weight'] = torch.sum(state_dict["conv1.weight"], dim=1, keepdim=True)
        # state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")

        model.load_state_dict(state_dict, strict=False)

    return model

