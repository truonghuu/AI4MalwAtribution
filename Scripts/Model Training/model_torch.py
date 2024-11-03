import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding,
                 activ='relu'):
        super(ResidualBlock, self).__init__()
        self.cnn1_0 = nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn1_0 = nn.BatchNorm1d(in_channel)
        self.cnn1_1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=False)
        self.bn1_1 = nn.BatchNorm1d(out_channel)

        # shortcut path
        self.cnn_sc = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,
                       stride=stride, padding=padding, bias=False)
        self.bn_sc = nn.BatchNorm1d(out_channel)
        if activ == 'softsign':
            self.activ = nn.Softsign()
        else:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        y0 = self.bn_sc(self.cnn_sc(x))
        y_ = self.activ(self.bn1_0(self.cnn1_0(x)))
        y1 = self.bn1_1(self.cnn1_1(y_))
        out = self.activ(y0 + y1)
        return out

class Net(nn.Module):
    """Each CNN block has two ways similar to ResNet:
        y = CNN_1(x) + CNN_1(CNN_0(x)
    where CNN_0 uses kernel=3, stride=1, padding=1 so it keeps the same input
    dimension, and has output_channels the same as input_channels. CNN_0 is
    just to learn some hidden features.
    CNN_1 compresses input dimension via stride>1, applying CNN_1 to CNN_0's
    output and x directly we hope can learn more features and combine them
    together to help backward signal propagate easily to the earlier layer.
    """
    def __init__(self, cfg):
        super().__init__()
        layers = []
        for i in range(len(cfg['conv_kernel_size'])):
            in_chan = 1 if i == 0 else cfg['out_channels'][i-1]
            out_chan = cfg['out_channels'][i]
            kernel_size = cfg['conv_kernel_size'][i]
            stride = cfg['conv_stride'][i]
            padding = cfg['conv_padding'][i]
            layers.append(ResidualBlock(in_chan, out_chan, kernel_size,
                            stride, padding,cfg['activ']))

        self.conv_net = nn.Sequential(*layers)

        drop_rates = cfg.get('dropout', [0]*(len(cfg['fc'])-1))
        self.dropout1 = nn.Dropout1d(drop_rates[0])
        layers= []
        for i in range(len(cfg['fc'])-1):
            in_fc = cfg['fc'][i]
            out_fc = cfg['fc'][i+1]
            if i == len(cfg['fc']) - 2: # last layer
                layers.extend(
                    [
                     #nn.BatchNorm1d(in_fc),
                     nn.Linear(in_fc, out_fc),
                     nn.Sigmoid()])
            else:
                layers.extend(
                    [
                     #nn.BatchNorm1d(in_fc),
                     nn.Linear(in_fc, out_fc),
                     nn.ReLU(inplace=True),
                     nn.Dropout1d(drop_rates[i+1])])
        self.clf = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_net(x)
        last_conv = torch.flatten(x, start_dim=1)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.clf(x)
        return x

class MLP(nn.Module):
    """Due to Tensorflow memory issue, reimplement in Pytorch.
    """
    def __init__(self, cfg):
        super(MLP, self).__init__()
        layers= []
        drop_rates = cfg.get('dropout', [0]*(len(cfg['fc'])-1))
        for i in range(len(cfg['fc'])-1):
            in_fc = cfg['fc'][i]
            out_fc = cfg['fc'][i+1]
            if out_fc == 1: # last layer
                layers.extend([nn.Linear(in_fc, out_fc),
                               nn.Sigmoid()])
            else:
                layers.extend(
                    [nn.Dropout(drop_rates[i]),
                    nn.Linear(in_fc, out_fc),
                    nn.ReLU(inplace=True)])
        self.clf = nn.Sequential(*layers)

    def forward(self, x):
        x = self.clf(x)
        x = torch.flatten(x, start_dim=1)
        return x


class Net1D(nn.Module):
    def __init__(self, cfg):
        super(Net1D, self).__init__()
        layers = []
        for i in range(len(cfg['conv_kernel_size'])):
            in_chan = 1 if i == 0 else cfg['out_channels'][i-1]
            out_chan = cfg['out_channels'][i]
            kernel_size = cfg['conv_kernel_size'][i]
            stride = cfg['conv_stride'][i]
            padding = cfg['conv_padding'][i]
            leaky = cfg.get('leaky_score', 0.1)
            if out_chan > 1:
                layers.extend(
                    [nn.Conv1d(in_chan, out_chan, kernel_size, stride, padding,
                            bias=False),
                    nn.BatchNorm1d(out_chan),
                    nn.LeakyReLU(leaky, inplace=True) if cfg.get('activ','relu') == 'relu' else nn.Softsign()
                    ])
            else:
                layers.extend(
                    [nn.Conv1d(in_chan, out_chan, kernel_size, stride,
                                        padding, bias=False),
                    nn.BatchNorm1d(out_chan)
                             ])

        self.conv_net = nn.Sequential(*layers)
        layers= []
        drop_rates = cfg.get('dropout', [0]*(len(cfg['fc'])-1))
        for i in range(len(cfg['fc'])-1):
            in_fc = cfg['fc'][i]
            out_fc = cfg['fc'][i+1]
            if out_fc == 1: # last layer
                layers.extend(
                    [nn.Dropout(drop_rates[-1]),
                    nn.Linear(in_fc, out_fc),
                    nn.Sigmoid()])
            else:
                layers.extend(
                    [nn.Dropout(drop_rates[i]),
                    nn.Linear(in_fc, out_fc),
                    nn.ReLU(inplace=True)])
        self.clf = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, start_dim=1)
        last_conv = x
        x = self.clf(x)
        return x

