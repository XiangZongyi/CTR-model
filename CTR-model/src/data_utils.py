import os
import pandas as pd
import numpy as np
import random
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

from utils import convert_sp_to_tensor


def create_dataloader(dataset, batch_size, training=False):
    if training:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class LoadDataset():
    def __init__(self, args):
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name

        self.train_inter_feat, self.test_inter_feat, inter_feat = self._load_train_test_feat()  
        self.user_feat = self._load_user_item_feat(type='user')
        self.item_feat = self._load_user_item_feat(type='item')

        self.user_num = inter_feat['user_id'].max() + 1
        self.item_num = inter_feat['item_id'].max() + 1

        self.train_user_items = {}  # {0:[1,2,3...], ...}
    
    def _load_train_test_feat(self):
        inter_feat = self.load_row_dataframe('inter')
        feat_path_train = os.path.join(self.dataset_path, self.dataset_name, "train.csv")
        feat_path_test = os.path.join(self.dataset_path, self.dataset_name, "test.csv")
        if os.path.isfile(feat_path_train) and os.path.isfile(feat_path_test):
            train_inter_feat = pd.read_csv(feat_path_train, sep='\t')
            test_inter_feat = pd.read_csv(feat_path_test, sep='\t')
            train_inter_feat['train_items'] = train_inter_feat['train_items'].\
                apply(lambda x: list(map(int, x[1:-1].split(','))))
            test_inter_feat['test_items'] = test_inter_feat['test_items'].\
                apply(lambda x: list(map(int, x[1:-1].split(','))))
        else:
            print(f'--------Preprocess interaction feature data---------------')
            train_inter_feat, test_inter_feat = split_inter_feat(inter_feat, self.args.data_split_ratio)
            train_inter_feat.to_csv(feat_path_train, sep='\t', index=False)
            test_inter_feat.to_csv(feat_path_test, sep='\t', index=False)
        return train_inter_feat, test_inter_feat, inter_feat
    
    def _load_user_item_feat(self, type=None):
        feat_path = os.path.join(self.dataset_path, self.dataset_name, f"{type}_onehot.csv")

        if not os.path.isfile(feat_path):
            print(f'--------Preprocess {type} feature data---------------')
            row_feat = self.load_row_dataframe(type)
            if type == 'user':
                onehot_feat = user_get_one_hot(row_feat, feat_path)
            elif type == 'item':
                onehot_feat = item_get_one_hot(row_feat, feat_path)
            else:
                raise ValueError("the value of type is error!")
        else:
            onehot_feat = pd.read_csv(feat_path, sep='\t')
            onehot_feat['onehot'] = onehot_feat['onehot'].\
                apply(lambda x: list(map(int, x[1:-1].split(','))))
        
        return onehot_feat
    
    def load_row_dataframe(self, dataset_type):
        if dataset_type in ['inter', 'user', 'item']:
            file_path = os.path.join(self.dataset_path, 
            self.dataset_name, f'{self.dataset_name}.{dataset_type}')
        else:
            return ValueError('dataset_type is error!')

        columns = []
        usecols = []
        dtype = {}
        with open(file_path, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split('\t'):
            field, ftype = field_type.split(':')
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == 'float' else str
        df = pd.read_csv(
            file_path, delimiter='\t', usecols=usecols, dtype=dtype
        )
        df.columns = columns

        # reset user(item) id from 1-943(1-1682) to 0-942(0-1681)
        if 'user_id' in columns and df['user_id'].min() =='1':
            df['user_id'] = df['user_id'].apply(lambda x: int(x) - 1)
        if 'item_id' in columns and df['item_id'].min() == '1':
            df['item_id'] = df['item_id'].apply(lambda x: int(x) - 1)
        return df
    
    def get_train_dataset(self):
        users, items, ratings = [], [], []
        row, col = [], []  # userd for creating interaction matrix
        for df_row in self.train_inter_feat.itertuples():
            user_id = getattr(df_row, 'user_id')
            pos_item = getattr(df_row, 'train_items')
            neg_sample_num = len(pos_item) * self.args.neg_sample_num
            neg_item = self.sample_negative(pos_item, neg_sample_num)

            users.extend([user_id] * len(pos_item))
            items.extend(pos_item)
            ratings.extend([1.0] * len(pos_item))
            users.extend([user_id] * len(neg_item))
            items.extend(neg_item)
            ratings.extend([0.0] * len(neg_item))

            row.extend([user_id] * len(pos_item))
            col.extend(pos_item)
            self.train_user_items[user_id] = pos_item
        inter_mat = self.create_inter_mat(row, col)
        train_dataset = TorchDataset(user=torch.LongTensor(users),
                                     item=torch.LongTensor(items),
                                     rating=torch.FloatTensor(ratings))
        return train_dataset, inter_mat


    def sample_negative(self, pos_item, sampling_num):
        neg_item = []
        for i in range(sampling_num):
            while True:
                negitem = random.choice(range(self.item_num))
                if negitem not in pos_item:
                    break
            neg_item.append(negitem)
        return neg_item

    def create_inter_mat(self, row, col):
        data = np.ones(len(row))

        mat = sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))  
        return mat 
    
    def get_test_dataset(self):
        users, items, ratings = [], [], []
        for df_row in self.test_inter_feat.itertuples():
            user_id = getattr(df_row, 'user_id')
            pos_item = getattr(df_row, 'test_items')
            pos_item_add_train = pos_item + self.train_user_items[user_id]
            neg_sample_num = len(pos_item) * self.args.neg_sample_num
            neg_item = self.sample_negative(pos_item_add_train, neg_sample_num)

            users.extend([user_id] * len(pos_item))
            items.extend(pos_item)
            ratings.extend([1.0] * len(pos_item))
            users.extend([user_id] * len(neg_item))
            items.extend(neg_item)
            ratings.extend([0.0] * len(neg_item))
        test_dataset = TorchDataset(user=torch.LongTensor(users),
                                    item=torch.LongTensor(items),
                                    rating=torch.FloatTensor(ratings))
        return test_dataset
    
    def get_feature_mat(self):
        user_feat_array = []
        for df_row in self.user_feat.itertuples():
            row_onehot = getattr(df_row, 'onehot')
            user_feat_array.append(row_onehot)
        user_feat_array = np.array(user_feat_array)
        user_feat_mat = sp.csr_matrix(user_feat_array, dtype=np.float32)

        item_feat_array = []
        for df_row in self.item_feat.itertuples():
            row_onehot = getattr(df_row, 'onehot')
            item_feat_array.append(row_onehot)
        item_feat_array = np.array(item_feat_array)
        item_feat_mat = sp.csr_matrix(item_feat_array, dtype=np.float32)

        # convert sparse matrix to torch tensor
        user_feat_mat = convert_sp_to_tensor(user_feat_mat).to(self.args.device)
        item_feat_mat = convert_sp_to_tensor(item_feat_mat).to(self.args.device)

        return user_feat_mat, item_feat_mat


def split_inter_feat(inter_feat, data_split_ratio):
    interact_status = inter_feat.groupby('user_id')['item_id'].apply(set).reset_index().rename(
    columns={'item_id': 'interacted_items'}
    )  # user-item_dic-interaction DataFrame: user, interacted_items

    # split train and test randomly by data_split_ratio
    interact_status['train_items'] = interact_status['interacted_items'].\
        apply(lambda x: set(random.sample(x, round(len(x) * data_split_ratio[0]))))
    interact_status['test_items'] = interact_status['interacted_items'] - interact_status['train_items']
    interact_status['train_items'] = interact_status['train_items'].apply(list)
    interact_status['test_items'] = interact_status['test_items'].apply(list)

    train_inter_feat = interact_status[['user_id', 'train_items']]
    test_inter_feat = interact_status[['user_id', 'test_items']] 
    return train_inter_feat, test_inter_feat


def user_get_one_hot(user_feat, save_path):
    occupation_list = []
    for row in user_feat.itertuples():
        occupations = getattr(row, 'occupation')
        for occupation in occupations.split(' '):
            if occupation not in occupation_list:
                occupation_list.append(occupation)
    random.shuffle(occupation_list)

    onehot = []
    for row in user_feat.itertuples():
        # age between 10-73(type: str)
        # 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79
        age_onehot = [0] * 7
        gender_onehot = [0] * 2
        occupation_onehot = [0] * len(occupation_list)

        age = getattr(row, 'age')
        gender = getattr(row, 'gender')
        occupation = getattr(row, 'occupation')

        age_onehot[int(age) // 10 - 1] = 1
        if gender == 'M':
            gender_onehot[0] = 1
        elif gender == 'F':
            gender_onehot[1] = 1
        occupation_onehot[occupation_list.index(occupation)] = 1

        age_onehot.extend(gender_onehot)
        age_onehot.extend(occupation_onehot)
        onehot.append(age_onehot)
    user_feat['onehot'] = onehot
    user_onehot = user_feat[['user_id', 'onehot']]
    user_onehot.to_csv(save_path, sep='\t', index=False)
    return user_onehot


def item_get_one_hot(item_feat, save_path):
    movie_class_list = []
    for row in item_feat.itertuples():
        movie_classes = getattr(row, '_4')
        for movie_class in movie_classes.split(' '):
            if movie_class not in movie_class_list:
                movie_class_list.append(movie_class)
    random.shuffle(movie_class_list)

    onehot = []
    for row in item_feat.itertuples():
        class_onehot = [0] * len(movie_class_list)
        movie_classes = getattr(row, '_4')
        for movie_class in movie_classes.split(' '):
            class_onehot[movie_class_list.index(movie_class)] = 1
        onehot.append(class_onehot)
    item_feat['onehot'] = onehot
    item_onehot = item_feat[['item_id', 'onehot']]
    item_onehot.to_csv(save_path, sep='\t', index=False)
    return item_onehot


class TorchDataset(Dataset):
    def __init__(self, user, item, rating):
        super(Dataset, self).__init__()

        self.user = user
        self.item = item
        self.rating = rating

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.rating[idx]
