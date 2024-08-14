import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class TM(Dataset):

    def __init__(self, args, mode):
        super(TM, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes
        self.k_shot = args.update_batch_size
        self.k_query = args.update_batch_size_eval
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/Gene_data/tab_train.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '{}/Gene_data/tab_test.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))
        self.classes_idx = list(self.data.keys())

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        support_x = np.zeros((self.args.meta_batch_size, self.set_size, 2866))
        query_x = np.zeros((self.args.meta_batch_size, self.query_size, 2866))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            for j in range(self.nb_classes):
                self.samples_idx = np.arange(self.data[self.choose_classes[j]].shape[0])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                    self.choose_classes[
                        j]][choose_samples[
                            :self.k_shot], ...]
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                    self.choose_classes[
                        j]][choose_samples[
                            self.k_shot:], ...]
                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return torch.FloatTensor(support_x), torch.LongTensor(support_y), torch.FloatTensor(query_x), torch.LongTensor(query_y)
