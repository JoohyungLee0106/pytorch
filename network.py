import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net_res3D_attention(nn.Module):
    def __init__(self):
        super(Net_res3D_attention, self).__init__()
        self.conv1= nn.Conv3d(1, 32, (1,7,7), padding=(0,3,3), stride=(1,2,2), bias=False)
        self.bn = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convblock21 = Convblock(32, 32, 1)
        self.convblock22 = Convblock(32, 32, 1)
        self.convblock31 = Convblock(32, 64, 2)
        self.convblock32 = Convblock(64, 64, 1)

        self.attention1 = Convblock(64, 64, 1)
        self.attention2 = nn.Conv3d(64, 1, (1,1,1), padding=0, stride=1, bias=True)

        self.convblock41 = Convblock(64, 128, 2)
        self.convblock42 = Convblock(128, 128, 1)
        self.convblock51 = Convblock(128, 256, 2)
        self.convblock52 = Convblock(256, 256, 1)

        # D*H*W
        # self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(256, 1)
        # self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 1)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, my_device):
        # print("Input: " + str(x.size()))
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.relu(out)
        out = self.convblock21(out)
        out = self.convblock22(out)
        out = self.convblock31(out)
        out = self.convblock32(out)
        attention_mask = self.attention1(out)
        attention_mask = self.attention2(attention_mask)
        print("attention_mask: "+str(attention_mask.size()))
        # 0 if negative (or 0), 1 if positive

        # Hard Attention
        # attention_mask_hard = F.relu(attention_mask).sign()
        # attention_mask_hard = attention_mask_hard.repeat(1, out.size(1), 1, 1, 1)        #
        # out = out * attention_mask_hard

        # Soft Attention
        attention_mask_soft = torch.where(attention_mask > 0.1, attention_mask, torch.zeros(attention_mask.size()).to(my_device))
        print("attention_mask_soft(1): " + str(attention_mask_soft.size()))
        attention_mask_soft = attention_mask_soft.repeat(1, out.size(1), 1, 1, 1)
        print("attention_mask_soft(2): " + str(attention_mask_soft.size()))
        out = out * attention_mask_soft
        print("out: " + str(out.size()))

        out = self.convblock41(out)
        out = self.convblock42(out)
        out = self.convblock51(out)
        out = self.convblock52(out)
        out = self.avgpool(out)
        out=out.view(-1, 256)
        # print("before fc: "+str(out.size()))
        # out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc(out)
        # out = nn.Sigmoid(out)
        return out, attention_mask

    # def cuda(self, device_number):

class Net_res3D_max(nn.Module):
    def __init__(self):
        super(Net_res3D_max, self).__init__()
        self.conv1= nn.Conv3d(1, 32, (1,7,7), padding=(0,3,3), stride=(1,2,2), bias=False)
        self.bn = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convblock21 = Convblock(32, 32, 1)
        self.convblock22 = Convblock(32, 32, 1)
        self.convblock31 = Convblock(32, 64, 2)
        self.convblock32 = Convblock(64, 64, 1)
        self.convblock41 = Convblock(64, 128, 2)
        self.convblock42 = Convblock(128, 128, 1)
        self.convblock51 = Convblock(128, 256, 2)
        self.convblock52 = Convblock(256, 256, 1)
        # D*H*W
        # self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(256, 1)
        # self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 1)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # print("Input: " + str(x.size()))
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.relu(out)
        out = self.convblock21(out)
        out = self.convblock22(out)
        out = self.convblock31(out)
        out = self.convblock32(out)
        out = self.convblock41(out)
        out = self.convblock42(out)
        out = self.convblock51(out)
        out = self.convblock52(out)
        out = self.avgpool(out)
        out=out.view(-1, 256)
        # print("before fc: "+str(out.size()))
        # out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc(out)
        # out = nn.Sigmoid(out)
        return out

class Net_res3D_avg(nn.Module):
    def __init__(self):
        super(Net_res3D_avg, self).__init__()
        self.conv1= nn.Conv3d(1, 32, (1,7,7), padding=(0,3,3), stride=(1,2,2), bias=False)
        self.bn = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convblock21 = Convblock(32, 32, 1)
        self.convblock22 = Convblock(32, 32, 1)
        self.convblock31 = Convblock(32, 64, 2)
        self.convblock32 = Convblock(64, 64, 1)
        self.convblock41 = Convblock(64, 128, 2)
        self.convblock42 = Convblock(128, 128, 1)
        self.convblock51 = Convblock(128, 256, 2)
        self.convblock52 = Convblock(256, 256, 1)
        # D*H*W
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        # self.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(256, 1)
        # self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 1)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, if_training):
        print("if_training: "+str(if_training))
        # print("Input: " + str(x.size()))
        out = self.conv1(x)
        # out = F.conv3d(x, )
        out = self.bn(out, if_training)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.convblock21(out, if_training)
        out = self.convblock22(out, if_training)
        out = self.convblock31(out, if_training)
        out = self.convblock32(out, if_training)
        out = self.convblock41(out, if_training)
        out = self.convblock42(out, if_training)
        out = self.convblock51(out, if_training)
        out = self.convblock52(out, if_training)
        out = self.avgpool(out)
        out=out.view(-1, 256)
        # print("before fc: "+str(out.size()))
        # out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc(out)
        # out = nn.Sigmoid(out)
        return out



class Convblock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(Convblock, self).__init__()

        if stride == 1:
            self.ifdimsame = True
        elif stride == 2:
            self.ifdimsame = False
            self.conv_downsample = nn.Conv3d(channel_in, channel_out, 1, stride = (1, 2, 2), bias=True)
        else:
            raise ValueError("stride should be 1 or 2")

        # kernel_size, stride, padding, dilation: (depth, height, width)
        self.conv1 = nn.Conv3d(channel_in, channel_in, 3, stride = (1, stride, stride), padding=(1,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(channel_in)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(channel_in, channel_out, 3, padding=(1,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(channel_out)
        self.relu2 = nn.ReLU()

    def forward(self, x, if_training):
        if self.ifdimsame:
            identity = x
            # print("identity size: " + str(identity.size()))
        else:
            # print("before conv_downsample: "+str(x.size()))
            identity = self.conv_downsample(x)
            # print("after conv_downsample: " + str(identity.size()))
        out = self.conv1(x)
        # print("after conv1: " + str(out.size()))
        out = self.bn1(out, track_running_stats=if_training)
        out = self.relu1(out)

        out = self.conv2(out)
        # print("after conv2: " + str(out.size()))
        out = self.bn2(out, track_running_stats=if_training)

        # print("after all conv: " + str(out.size()))
        out += identity

        out = self.relu2(out)
        # print("=====================")

        return out




class Convblock_deprecated(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Convblock, self).__init__()

        if channel_in == channel_out:
            self.ifdimsame = True
            self.conv1 = nn.Conv3d(channel_in, channel_in, 3, stride = 1,  padding=(1,1,1),bias=False)
        else:
            self.ifdimsame = False
            self.conv_downsample = nn.Conv3d(channel_in, channel_out, 1, stride = (1, 2, 2), bias=True)
            # kernel_size, stride, padding, dilation: (depth, height, width)
            self.conv1 = nn.Conv3d(channel_in, channel_in, 3, stride = (1, 2, 2), padding=(1,1,1), bias=False)

        self.bn1 = nn.BatchNorm3d(channel_in)
        self.conv2 = nn.Conv3d(channel_in, channel_out, 3, padding=(1,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(channel_out)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        if self.ifdimsame:
            identity = x
            # print("identity size: " + str(identity.size()))
        else:
            # print("before conv_downsample: "+str(x.size()))
            identity = self.conv_downsample(x)
            # print("after conv_downsample: " + str(identity.size()))
        out = self.conv1(x)
        # print("after conv1: " + str(out.size()))
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        # print("after conv2: " + str(out.size()))
        out = self.bn2(out)

        # print("after all conv: " + str(out.size()))
        out += identity

        out = self.relu2(out)
        # print("=====================")
        return out