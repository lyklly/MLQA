import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Beta
import numpy as np
from kornia import augmentation as K
device = torch.device('cuda:0')
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class FCNet(nn.Module):
    def __init__(self, args, x_dim, hid_dim, dropout=0.2):
        super(FCNet, self).__init__()
        self.args = args
        self.net = nn.Sequential(
            self.fc_block(x_dim, hid_dim, dropout),
            self.fc_block(hid_dim, hid_dim, dropout),
        )
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.logits = nn.Linear(self.hid_dim, self.args.num_classes)

    def fc_block(self, in_features, out_features, dropout):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def functional_fc_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):

        x = F.linear(x, weights, biases)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        return x

    def mixup_data(self, xs, ys, xq, yq, lam = None):
        query_size = xq.shape[0]
        if len(xq.shape) == 2:
            xs = xs.repeat(int(xq.shape[0] // xs.shape[0]), 1)
        else:
            xs = xs.repeat(int(xq.shape[0] // xs.shape[0]), 1, 1, 1)
        ys = ys.repeat(int(yq.shape[0] // ys.shape[0]))
        shuffled_index = torch.randperm(query_size)

        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        if lam == None:
            lam = self.dist.sample().to(device)
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, yq, ys, lam

    def forward(self, x):
        hidden = self.net(x)
        hidden = self.logits(hidden)
        return hidden

    def functional_forward(self, x, weights, is_training=True):
        for block in range(2):
            x = self.functional_fc_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)
        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])
        return x

    def channel_shuffle(self, hidden, label, shuffle_dict, shuffle_channel_id, shuffle = True):
        if hidden.shape[-1] == 2866:
            concept_idx = [0, 528, 968, 1408, 1936, 2376, 2866]     
        else:
            concept_idx = [0, 12, 22, 32, 44, 54, 64]

        new_data = []

        start = concept_idx[shuffle_channel_id]
        end = concept_idx[shuffle_channel_id + 1]

        for i in range(self.args.num_classes):
            cur_class_1 = hidden[label == i]
            cur_class_2 = hidden[label == shuffle_dict[i]]

            new_data.append(
                torch.cat((cur_class_1[:, :start], cur_class_2[:, start:end], cur_class_1[:, end:]), dim=1))

        new_data = torch.cat(new_data, dim=0)

        indexes = torch.randperm(new_data.shape[0])
        new_label = label
        if shuffle == True:
            new_data = new_data[indexes]
            new_label = label[indexes]

        return new_data, new_label, indexes


    def functional_forward_cf(self, hidden, label, sel_layer, shuffle_list, shuffle_channel_id, weights,
                                           is_training=True):

        label_new = label

        for layer in range(2):
            if layer == sel_layer:
                hidden, label_new, _ = self.channel_shuffle(hidden, label, shuffle_list, shuffle_channel_id)

            hidden = self.functional_fc_block(hidden, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
        x = F.linear(hidden, weights['logits.weight'], weights['logits.bias'])

        return x, label_new


    def our_mix(self, xs, xq, lam):
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, lam

    def augmentation(self, x1s, y1s, x2s, y2s, lam_mix):
        da_pool = ['mp']
        methods = random.sample(da_pool, 1)[0]
        if methods == 'cm':
            x1s, lam_mix = self.mix_data(x1s, x2s, lam_mix)
        return x1s, lam_mix

    def MLQA(self, x1s, y1s, x2s, y2s, x1q, y1q, x2q, y2q, lam_mix, task_2_shuffle_id, shuffle_list, shuffle_channel_id):

        kshot = int(x1s.shape[0] // self.args.num_classes)
        task_2_shuffle = np.array(
            [np.arange(kshot) + task_2_shuffle_id[idx] * kshot for idx in
            range(self.args.num_classes)]).flatten()

        x2s = x2s[task_2_shuffle]
        y2s = y2s[task_2_shuffle]
        x1s, lamda = self.augmentation(x1s, y1s, x2s, y2s, lam_mix)
        
        x1s, y1s, indexes = self.channel_shuffle(x1s, y1s, shuffle_list, shuffle_channel_id)
        x1q, y1q, _ = self.channel_shuffle(x1q, y1q, shuffle_list, shuffle_channel_id)
        x2q, y2q, _ = self.channel_shuffle(x2q, y2q, shuffle_list, shuffle_channel_id)

        x1q, query_label, support_label, lamda = self.mixup_data(x2q, y2q, x1q, y1q)
        x1q = torch.cat((x1q, x1s))
        q_label = torch.cat((query_label, y1s))
        support_label = torch.cat((support_label, y1s))

        return x1q, q_label, support_label, lamda

    def functional_forward_MLQA(self, x1s, y1s, x2s, y2s, x1q, y1q, x2q, y2q, sel_layer, shuffle_list, shuffle_channel_id, weights, is_training=True):
        lam_mix = self.dist.sample().to(x1s.device)
        task_2_shuffle_id = np.arange(self.args.num_classes)
        np.random.shuffle(task_2_shuffle_id)

        flag = 0

        for layer in range(2):
            if layer == sel_layer:
                x1q, new_label, support_label, lamda = self.MLQA(x1s, y1s, x2s, y2s, x1q, y1q, x2q, y2q,
                                                                        lam_mix,
                                                                        task_2_shuffle_id,
                                                                        shuffle_list,
                                                                        shuffle_channel_id)

                flag = 1

            if not flag:
                x1s = self.functional_fc_block(x1s, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
                x2s = self.functional_fc_block(x2s, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
                x2q = self.functional_fc_block(x2q, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
            x1q = self.functional_fc_block(x1q, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)
        
        x = F.linear(x1q, weights['logits.weight'], weights['logits.bias'])

        return x, new_label, support_label, lamda

