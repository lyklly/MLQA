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

class Conv_Standard(nn.Module):
    def __init__(self, args, x_dim, hid_dim, z_dim, final_layer_size):
        super(Conv_Standard, self).__init__()
        self.args = args
        self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                 self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim), Flatten())
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(final_layer_size, self.args.num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):

        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        return x

    def mixup_data(self, xs, ys, xq, yq, lam = None):
        query_size = xq.shape[0]
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
        x = self.net(x)
        x = torch.mean(x, dim=(2, 3))

        return self.logits(x)

    def functional_forward(self, x, weights, is_training=True):
        for block in range(4):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)
        x = torch.mean(x, dim=(2, 3))
        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])
        return x

    def channel_shuffle(self, hidden, label, shuffle_dict, shuffle_channel_id, shuffle = True):

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

        for layer in range(4):
            if layer == sel_layer:
                hidden, label_new, _ = self.channel_shuffle(hidden, label, shuffle_list, shuffle_channel_id)

            hidden = self.functional_conv_block(hidden, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
        hidden = torch.mean(hidden, dim=(2, 3))
        x = F.linear(hidden, weights['logits.weight'], weights['logits.bias'])

        return x, label_new

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam.cpu())
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def mix_data(self, xs, xq, lam):
        mixed_x = xq.clone()
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(xq.size(), lam)

        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = xs[:, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xq.size()[-1] * xq.size()[-2]))
        lam = torch.Tensor([lam]).to(device)
        return mixed_x, lam
    
    def our_mix(self, x1s, x2s, lam_mix):
        mixed_x = x1s.clone()
        mixed_x = lam_mix * x1s + (1 - lam_mix) * x2s

        return mixed_x

    def random_erase(self, data):
        rec_er = K.RandomErasing(scale=(.02, .4), ratio=(0.3, 1/.3), p=0.5)
        erase_data = data.clone()
        out = rec_er(erase_data.view([-1] + list(data.shape[-3:])))

        return out.view(data.shape)

    def tlrot(self, data, label):
        clone_data = data.clone()
        for j in range(self.args.num_classes):
            k = random.sample([0, 0, 0, 1, 2, 3], 1)[0]
            clone_data[label == j] = torch.rot90(clone_data[label == j], k, [2, 3])

        return clone_data

   def augmentation(self, x1s, y1s, x2s, y2s, lam_mix = 0, da_pool = []):
        if lam_mix == 0:
            lam_mix = self.dist.sample().to(x1s.device)
        if len(da_pool) == 0:
            da_pool = ['cm', 'mp', 'cm + re', 'cm + tlr', 'mp + re', 'mp + tlr', 'cm + re + tlr', 'mp + re + tlr']
        methods = random.sample(da_pool, 1)[0]
        if methods == 'cm':
            x2s, lam_mix = self.mix_data(x2s, x1s, lam_mix)
        elif methods == 'mp':
            lam_mix = lam_mix / 2
            x1s = self.our_mix(x1s, x2s, lam_mix)            
        elif methods == 're':
            x1s = self.random_erase(x1s)
        elif methods == 'tlr':
            x1s = self.tlrot(x1s, y1s)
        elif methods == 'cm + re':
            x1s = self.random_erase(x1s)
            x2s = self.random_erase(x2s)
            x1s, lam_mix = self.mix_data(x2s, x1s, lam_mix)
        elif methods == 'cm + tlr':
            x1s = self.tlrot(x1s, y1s)
            x2s = self.tlrot(x2s, y2s)
            x1s, lam_mix = self.mix_data(x2s, x1s, lam_mix)
        elif methods == 'mp + re':
            lam_mix = lam_mix / 2
            x1s = self.random_erase(x1s)
            x2s = self.random_erase(x2s)
            x1s = self.our_mix(x1s, x2s, lam_mix) 
        elif methods == 'mp + tlr':
            lam_mix = lam_mix / 2
            x1s = self.tlrot(x1s, y1s)
            x2s = self.tlrot(x2s, y2s)
            x1s = self.our_mix(x1s, x2s, lam_mix) 
        elif methods == 're + tlr':
            x1s = self.tlrot(x1s, y1s)
            x1s = self.random_erase(x1s)
        elif methods == 'cm + re + tlr':
            x1s = self.tlrot(x1s, y1s)
            x1s = self.random_erase(x1s)
            x2s = self.tlrot(x2s, y2s)
            x2s = self.random_erase(x2s)
            x1s, lam_mix = self.mix_data(x2s, x1s, lam_mix)
        elif methods == 'mp + re + tlr': 
            lam_mix = lam_mix / 2
            x1s = self.tlrot(x1s, y1s)
            x1s = self.random_erase(x1s)
            x2s = self.tlrot(x2s, y2s)
            x2s = self.random_erase(x2s)
            x1s = self.our_mix(x1s, x2s, lam_mix)
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

        query_label = y1q
        support_label = y2q
        x1q, lamda = self.mix_data(x2q, x1q, lam_mix)
        x1q = torch.cat((x1q, x1s))
        q_label = torch.cat((query_label, y1s))
        support_label = torch.cat((support_label, y1s))

        return x1q, q_label, support_label, lamda

    def functional_forward_MLQA(self, x1s, y1s, x2s, y2s, x1q, y1q, x2q, y2q, sel_layer, shuffle_list, shuffle_channel_id, weights, is_training=True):
        lam_mix = self.dist.sample().to(x1s.device)
        task_2_shuffle_id = np.arange(self.args.num_classes)
        np.random.shuffle(task_2_shuffle_id)

        flag = 0

        for layer in range(4):
            if layer == sel_layer:
                x1q, new_label, support_label, lamda = self.MLQA(x1s, y1s, x2s, y2s, x1q, y1q, x2q, y2q,
                                                                        lam_mix,
                                                                        task_2_shuffle_id,
                                                                        shuffle_list,
                                                                        shuffle_channel_id)

                flag = 1

            if not flag:
                x1s = self.functional_conv_block(x1s, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
                x2s = self.functional_conv_block(x2s, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
                x2q = self.functional_conv_block(x2q, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
            x1q = self.functional_conv_block(x1q, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)

        x1q = torch.mean(x1q, dim=(2, 3))
        x = F.linear(x1q, weights['logits.weight'], weights['logits.bias'])

        return x, new_label, support_label, lamda
