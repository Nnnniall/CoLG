import torch
from torch import nn
from torch.nn import init
import numpy as np
import math
import option

torch.set_default_tensor_type('torch.FloatTensor')
args = option.parser.parse_args()


# 初始化参数
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)  # Xavier 初始化
        if m.bias is not None:
            m.bias.data.fill_(0)


# Disentangled Non-Local Block
class _NonLocalNd_nowd(nn.Module):
    def __init__(self, dim, inplanes, planes, downsample, lr_mult, use_out, out_bn, whiten_type, weight_init_scale,
                 with_gc, with_nl, eps, nowd):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['in', 'in_nostd', 'ln', 'ln_nostd', 'fln', 'fln_nostd'] # all without affine, in == channel whiten
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(_NonLocalNd_nowd, self).__init__()

        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None

        if with_nl:
            self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.weight_init_scale = weight_init_scale
        self.with_gc = with_gc
        self.with_nl = with_nl
        self.nowd = nowd
        self.eps = eps

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)
        self.reset_weight_and_weight_decay()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def reset_weight_and_weight_decay(self):
        if self.with_nl:
            init.normal_(self.conv_query.weight, 0, 0.01 * self.weight_init_scale)
            init.normal_(self.conv_key.weight, 0, 0.01 * self.weight_init_scale)
            if 'nl' in self.nowd:
                self.conv_query.weight.wd = 0.0
                self.conv_query.bias.wd = 0.0
                self.conv_key.weight.wd = 0.0
                self.conv_key.bias.wd = 0.0
        if self.with_gc and 'gc' in self.nowd:
            self.conv_mask.weight.wd = 0.0
            self.conv_mask.bias.wd = 0.0
        if 'value' in self.nowd:
            self.conv_value.weight.wd = 0.0
            # self.conv_value.bias.wd=0.0

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        value = self.conv_value(input_x)
        value = value.view(value.size(0), value.size(1), -1)

        out_sim = None

        if self.with_nl:
            # [N, C', T, H, W]
            query = self.conv_query(x)
            # [N, C', T, H', W']
            key = self.conv_key(input_x)

            # [N, C', H x W]
            query = query.view(query.size(0), query.size(1), -1)
            # [N, C', H' x W']
            key = key.view(key.size(0), key.size(1), -1)

            if 'in_nostd' in self.whiten_type:
                key_mean = key.mean(2).unsqueeze(2)
                query_mean = query.mean(2).unsqueeze(2)
                key = key - key_mean
                query = query - query_mean
            elif 'in' in self.whiten_type:
                key_mean = key.mean(2).unsqueeze(2)
                query_mean = query.mean(2).unsqueeze(2)
                key = key - key_mean
                query = query - query_mean
                key_var = key.var(2).unsqueeze(2)
                query_var = query.var(2).unsqueeze(2)
                key = key / torch.sqrt(key_var + self.eps)
                query = query / torch.sqrt(query_var + self.eps)
            elif 'ln_nostd' in self.whiten_type:
                key_mean = key.view(key.shape[0], -1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(query.shape[0], -1).mean(1).unsqueeze(1).unsqueeze(2)
                key = key - key_mean
                query = query - query_mean
            elif 'ln' in self.whiten_type:
                key_mean = key.view(key.shape[0], -1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(query.shape[0], -1).mean(1).unsqueeze(1).unsqueeze(2)
                key = key - key_mean
                query = query - query_mean
                key_var = key.view(key.shape[0], -1).var(1).unsqueeze(1).unsqueeze(2)
                query_var = query.view(query.shape[0], -1).var(1).unsqueeze(1).unsqueeze(2)
                key = key / torch.sqrt(key_var + self.eps)
                query = query / torch.sqrt(query_var + self.eps)
            elif 'fln_nostd' in self.whiten_type:
                key_mean = key.view(1, -1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(1, -1).mean(1).unsqueeze(1).unsqueeze(2)
                key = key - key_mean
                query = query - query_mean
            elif 'fln' in self.whiten_type:
                key_mean = key.view(1, -1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(1, -1).mean(1).unsqueeze(1).unsqueeze(2)
                key = key - key_mean
                query = query - query_mean
                key_var = key.view(1, -1).var(1).unsqueeze(1).unsqueeze(2)
                query_var = query.view(1, -1).var(1).unsqueeze(1).unsqueeze(2)
                key = key / torch.sqrt(key_var + self.eps)
                query = query / torch.sqrt(query_var + self.eps)

            # [N, T x H x W, T x H' x W']
            sim_map = torch.bmm(query.transpose(1, 2), key)
            ### cancel temp and scale
            if 'nl' not in self.nowd:
                sim_map = sim_map / self.scale
            sim_map = self.softmax(sim_map)

            # [N, T x H x W, C']
            out_sim = torch.bmm(sim_map, value.transpose(1, 2))
            # [N, C', T x H x W]
            out_sim = out_sim.transpose(1, 2)
            # [N, C', T,  H, W]
            out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
            out_sim = self.gamma * out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0, 2, 1))
            if out_sim is not None:
                out_sim = out_sim + out_gc
            else:
                out_sim = out_gc

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)

        out = out_sim + residual

        return out


class NonLocal1d_nowd(_NonLocalNd_nowd):
    def __init__(self, inplanes, planes, downsample=True, lr_mult=None, use_out=False, out_bn=False,
                 whiten_type=['in_nostd'], weight_init_scale=1.0, with_gc=False, with_nl=True, eps=1e-5, nowd=['nl']):
        super(NonLocal1d_nowd, self).__init__(dim=1, inplanes=inplanes, planes=planes, downsample=downsample,
                                              lr_mult=lr_mult, use_out=use_out, out_bn=out_bn,
                                              whiten_type=whiten_type, weight_init_scale=weight_init_scale,
                                              with_gc=with_gc, with_nl=with_nl, eps=eps, nowd=nowd)


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.division = 2048 // len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(512 // self.division)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512 // self.division)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512 // self.division)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048 // self.division, out_channels=512 // self.division, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048 // self.division, out_channels=2048 // self.division, kernel_size=3,
                      stride=1, padding=1, bias=False),  # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048 // self.division),
            # nn.dropout(0.7)
        )

        self.non_local = NonLocal1d_nowd(512 // self.division, 512 // (self.division * 2), downsample=False,
                                         whiten_type=['in_nostd'], weight_init_scale=1.0, with_gc=True, with_nl=True,
                                         nowd=['nl'], use_out=False, out_bn=False)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)

        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)  # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)

        return out


class Memory(nn.Module):
    def __init__(self, memory_size, key_dim, batch_size, ncrops):
        super(Memory, self).__init__()
        self.dim = key_dim
        self.nums = memory_size
        self.weight = nn.Parameter(torch.empty(memory_size, key_dim))
        self.bias = None
        self.sig = nn.Sigmoid()
        self.reset_parameters()
        self.batch_size = batch_size
        self.ncrops = ncrops

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.weight) / (self.dim ** 0.5))
        temporal_att = torch.topk(attention, self.nums // 16 + 1, dim=-1)[0].mean(-1)
        augment = torch.einsum('btk,kd->btd', attention[0:self.batch_size * self.ncrops], self.weight)
        return temporal_att, augment


class Model_mem(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model_mem, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.division = 2048 // n_features
        # 时间学习
        self.Aggregate = Aggregate(len_feature=2048 // self.division)
        # 多层感知机
        self.fc1 = nn.Linear(n_features, 1024 // self.division)
        self.fc2 = nn.Linear(1024 // self.division, 512 // self.division)
        self.fc3 = nn.Linear(512 // self.division, 1024 // self.division)
        self.fc4 = nn.Linear(1024 // self.division, n_features)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

        self.memory = Memory(memory_size=args.mem_size, key_dim=512 // self.division, batch_size=batch_size, ncrops=args.plot_freq)

    def forward(self, inputs, train):

        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        out = self.Aggregate(out)

        out = self.drop_out(out)

        features = self.relu(self.fc1(out))
        features = self.relu(self.fc2(features))

        if train:

            mem_score, updated_memory = self.memory(features)

            logits = 1 - mem_score

            nor_features = inputs.view(bs * ncrops, t, f)[0:self.batch_size * ncrops]

            de_features = self.relu(self.fc3(updated_memory))
            de_features = self.relu(self.fc4(de_features))

            return logits, nor_features, de_features, out
        else:
            mem_score, updated_memory = self.memory(features)
            logits = 1 - mem_score
            logits = logits.view(bs, ncrops, -1)
            logits = logits.mean(1).unsqueeze(dim=2)
            return logits
