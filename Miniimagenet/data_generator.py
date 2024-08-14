import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class Share_miniImagenet(Dataset):

    def __init__(self, args, mode):
        super(Share_miniImagenet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

        self.fixed_class_id={0: [58, 44, 39, 26, 45, 13, 23, 33, 34, 50, 17, 6, 28], 1: [32, 52, 38, 12, 43, 54, 1, 56, 25, 59, 5, 37, 16],
     2: [55, 11, 14, 41, 2, 61, 49, 31, 57, 46, 22, 62, 51], 3: [63, 4, 20, 21, 10, 36, 48, 18, 53, 3, 15, 24, 0],
     4: [9, 60, 42, 8, 7, 19, 47, 30, 35, 40, 29, 27]}

    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            if self.mode == 'train':
                self.choose_classes = []
                for j in range(self.nb_classes):
                    if j==4:
                        self.choose_classes.append(self.fixed_class_id[j][np.random.randint(12)])
                    else:
                        self.choose_classes.append(self.fixed_class_id[j][np.random.randint(13)])
            else:
                self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            for j in range(self.nb_classes):
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                    self.choose_classes[
                        j], choose_samples[
                            :self.k_shot], ...]
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                    self.choose_classes[
                        j], choose_samples[
                            self.k_shot:], ...]
                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class MiniImagenet(Dataset):

    def __init__(self, args, mode):
        super(MiniImagenet, self).__init__()
        self.args = args
        self.task_num = args.meta_batch_size
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/unmini/mini_imagenet_train.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '{}/unmini/mini_imagenet_test.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.all_train_classes = np.array([26, 59, 14, 16, 17, 52,  8, 39, 46, 32, 20, 57, 34, 25, 63, 31, 30,
       40,  0, 43,  7, 33, 12,  6, 22, 23, 49, 50, 15, 13, 51, 10, 24, 27,
       47, 55,  9,  5, 18, 36, 44, 35,  4, 21, 61, 42, 11,  3, 45, 58, 60,
       56,  1, 28, 48, 54, 37, 19, 62, 41, 38,  2, 53, 29])
        self.num_train_use_class = int(64*self.args.ratio)

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

    def __len__(self):
        return self.args.metatrain_iterations*self.task_num

    def __getitem__(self, index):
        if self.mode == 'train':
            self.classes_idx = self.all_train_classes[:self.num_train_use_class]
        else:
            self.classes_idx = np.arange(self.data.shape[0])

        # ipdb.set_trace()
        self.samples_idx = np.arange(self.data.shape[1])
        support_x = torch.FloatTensor(torch.zeros((self.task_num, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.task_num, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.task_num, self.set_size])
        query_y = np.zeros([self.task_num, self.query_size])

        for meta_batch_id in range(self.task_num):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            for j in range(self.nb_classes):
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                    self.choose_classes[
                        j], choose_samples[
                            :self.k_shot], ...]
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                    self.choose_classes[
                        j], choose_samples[
                            self.k_shot:], ...]
                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

