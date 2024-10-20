import numpy as np
import scipy.sparse as sp
import os
from utility.Tool import randint_choice, df_to_positive_dict, save_dict_to_file, load_dict_from_file, csr_to_user_dict
import pandas as pd
from time import time
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
KEEP_CONTEXT = {
    'yelp': ['c_city', 'c_year', 'c_month', 'c_DoW', 'c_last'],
    'yelp-oh': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'amazon-book': ['c_year', 'c_month', 'c_day', 'c_DoW', 'c_last'],
    'huawei-blog': ['c_year', 'c_month', 'c_day', 'c_hourGap'],
    'lfm': ['c_year', 'c_month', 'c_day', 'c_DoW', 'c_last']
}
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)

# def read_cf_new():
#     # reading rating file
#     rating_file = 'data/' + args.dataset + '/ratings_final'
#     if os.path.exists(rating_file + '.npy'):
#         rating_np = np.load(rating_file + '.npy')
#     else:
#         rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
#         np.save(rating_file + '.npy', rating_np)
#
#     # rating_np_origin = rating_np
#     # rating_np_label = rating_np.take([2], axis=1)
#     # indix_click = np.where(rating_np_label == 1)
#     # rating_np = rating_np.take(indix_click[0], axis=0)
#     # rating_np = rating_np.take([0, 1], axis=1)
#
#     test_ratio = 0.2
#     n_ratings = rating_np.shape[0]
#     eval_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
#     left = set(range(n_ratings)) - set(eval_indices)
#     # test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
#     train_indices = list(left)
#
#     train_data = rating_np[train_indices]
#     eval_data = rating_np[eval_indices]
#     # test_data = rating_np[test_indices]
#
#     train_rating = rating_np[train_indices]
#     ui_adj = generate_ui_adj(rating_np, train_rating)
#     return train_data, eval_data, ui_adj

def generate_ui_adj(rating, train_rating):
    #ui_adj = sp.dok_matrix((n_user, n_item), dtype=np.float32)
    n_user, n_item = len(set(rating[:, 0])), len(set(rating[:, 1]))
    ui_adj_orign = sp.coo_matrix(
        (train_rating[:, 2], (train_rating[:, 0], train_rating[:, 1])), shape=(n_user, n_item)).todok()

    # ui_adj = sp.dok_matrix((n_user+n_item, n_user+n_item), dtype=np.float32)
    # ui_adj[:n_user, n_user:] = ui_adj_orign
    # ui_adj[n_user:, :n_user] = ui_adj_orign.T
    ui_adj = sp.bmat([[None, ui_adj_orign],
                    [ui_adj_orign.T, None]], dtype=np.float32)
    ui_adj = ui_adj.todok()
    print('already create user-item adjacency matrix', ui_adj.shape)
    return ui_adj

def remap_item(train_data, eval_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1])) + 1

    # eval_data_label = eval_data.take([2], axis=1)
    # indix_click = np.where(eval_data_label == 1)
    # eval_data = eval_data.take(indix_click[0], axis=0)
    #
    # eval_data = eval_data.take([0, 1], axis=1)
    # train_data = train_data.take([0, 1], axis=1)
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in eval_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes
    inverse_r = True
    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # norm_mat_list[0] = norm_mat_list[0].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list


class GivenData(object):
    def __init__(self, dataset_name, path, data_format, separator, logger):
        self.dataset_name = dataset_name
        self.path = path
        self.data_format = data_format
        self.separator = separator
        self.logger = logger

    def load_data(self):
        # begin KGIN data
        print('reading train and test user-item set ...')
        train_cf = read_cf(self.path + 'train.txt')
        test_cf = read_cf(self.path + 'test.txt')
        # train_cf, eval_cf, ui_adj = read_cf_new()
        remap_item(train_cf, test_cf)

        print('combinating train_cf and kg data ...')
        triplets = read_triplets(self.path + 'kg_final.txt')

        print('building the graph ...')
        graph, relation_dict = build_graph(train_cf, triplets)

        print('building the adj mat ...')
        adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

        n_params = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'n_entities': int(n_entities),
            'n_nodes': int(n_nodes),
            'n_relations': int(n_relations)
        }
        user_dict = {
            'train_user_set': train_user_set,
            'test_user_set': test_user_set
        }




        side_info, all_data_dict = None, None
        self.logger.info("Loading interaction records from folder: %s "% (self.path))

        train_data = pd.read_csv(self.path + "train.dat", sep=self.separator[0])
        test_data = pd.read_csv(self.path + "test.dat", sep=self.separator[0])
        userid_dict = load_dict_from_file(self.path + 'userid_dict.txt')
        itemid_dict = load_dict_from_file(self.path + 'itemid_dict.txt')
        # self.logger.info('Loading full testset')

        all_data = pd.concat([train_data, test_data])

        num_users = len(userid_dict)
        num_items = len(itemid_dict) 
        num_valid_items = all_data["item_id"].max() + 1

        num_train = len(train_data["user_id"])
        num_test = len(test_data["user_id"])
        
        train_matrix = sp.csr_matrix(([1] * num_train, (train_data["user_id"], train_data["item_id"])), shape=(num_users, num_valid_items))
        test_matrix = sp.csr_matrix(([1] * num_test, (test_data["user_id"], test_data["item_id"])),  shape=(num_users, num_valid_items))
        
        if self.data_format == 'UIC':
            side_info, side_info_stats, all_data_dict = {}, {}, {}
            column = all_data.columns.values.tolist()
            context_column = column[2].split(self.separator[1])
            # print(self.dataset_name.lower())
            user_feature_column = column[-2].split(self.separator[1]) if 'yelp-oh' in self.dataset_name.lower() else None
            # PING Change Part
            item_feature_column = column[-1].split(self.separator[1]) if 'yelp-oh' in self.dataset_name.lower() else None

            keep_context = KEEP_CONTEXT[self.dataset_name.lower()]
            new_context_column = '-'.join(keep_context)
            # print(context_column)
            # print(new_context_column)
            all_data[context_column] = all_data[all_data.columns[2]].str.split(self.separator[1], expand=True)
            all_data[new_context_column] = all_data[keep_context].apply('-'.join, axis=1)
            # map context to id
            unique_context = all_data[new_context_column].unique()
            context2id = pd.Series(data=range(len(unique_context)), index=unique_context)
            # contextids = context2id.to_dict()
            all_data["context_id"] = all_data[new_context_column].map(context2id)
            train_data = all_data.iloc[:num_train, :]
            test_data = all_data.iloc[num_train:, :]

            if user_feature_column:
                user_feature = all_data.drop_duplicates(["user_id", '-'.join(user_feature_column)])
                user_feature = user_feature[["user_id", '-'.join(user_feature_column)]]
                user_feature[user_feature_column] = user_feature[user_feature.columns[-1]].str.split(self.separator[1], expand=True)
                user_feature.drop(user_feature.columns[[1]], axis=1, inplace=True)
            else:
                user_feature = None
            # PING Change Part
            # has no item feature
            if item_feature_column:
                item_feature = all_data.drop_duplicates(["item_id", '-'.join(item_feature_column)])
                item_feature = item_feature[["item_id", '-'.join(item_feature_column)]]
                item_feature[item_feature_column] = item_feature[item_feature.columns[-1]].str.split(
                    self.separator[1], expand=True)
                item_feature.drop(item_feature.columns[[1]], axis=1, inplace=True)
            else:
                item_feature = None
            # item_feature = all_data.drop_duplicates(["item_id", '-'.join(item_feature_column)]) item_feature =
            # item_feature[["item_id", '-'.join(item_feature_column)]] item_feature[item_feature_column] =
            # item_feature[item_feature.columns[-1]].str.split(self.separator[1], expand=True) item_feature.drop(
            # item_feature.columns[[1]], axis=1, inplace=True)

            context_feature = all_data.drop_duplicates(["context_id", new_context_column])[["context_id", new_context_column]]
            context_feature[keep_context] = context_feature[context_feature.columns[-1]].str.split(self.separator[1], expand=True)
            context_feature.drop(context_feature.columns[[1]], axis=1, inplace=True)
            if user_feature_column:
                side_info['user_feature'] = user_feature.set_index('user_id').astype(int)
                side_info_stats['num_user_features'] = side_info['user_feature'][user_feature_column[-1]].max() + 1
                side_info_stats['num_user_fields'] = len(user_feature_column)
            else:
                side_info['user_feature'] = None
                side_info_stats['num_user_features'] = 0
                side_info_stats['num_user_fields'] = 0
            # PING Change Part
            # has no item feature
            if item_feature_column:
                side_info['item_feature'] = item_feature.set_index('item_id').astype(int)
                side_info_stats['num_item_features'] = side_info['item_feature'][item_feature_column[-1]].max() + 1
                side_info_stats['num_item_fields'] = len(item_feature_column)
            else:
                side_info['item_feature'] = None
                side_info_stats['num_item_features'] = 0
                side_info_stats['num_item_fields'] = 0
            # side_info['item_feature'] = item_feature.set_index('item_id').astype(int)
            side_info['context_feature'] = context_feature.set_index('context_id').astype(int)
            # side_info_stats['num_item_features'] = side_info['item_feature'][item_feature_column[-1]].max() + 1
            # side_info_stats['num_item_fields'] = len(item_feature_column)
            if self.dataset_name.lower()=='huawei-blog':
                side_info_stats['num_context_features'] = side_info['context_feature'][
                                                              keep_context[-1]].max() + 1
            else:
                side_info_stats['num_context_features'] = side_info['context_feature'][
                                                              keep_context[-2]].max() + 1 + num_items
            side_info_stats['num_context_fields'] = len(keep_context)
            self.logger.info("\n" + "\n".join(["{}={}".format(key, value) for key, value in side_info_stats.items()]))
            self.logger.info("context feature name: " + ",".join([f.replace('c_', '') for f in keep_context]))
            all_data_dict['train_data'] = train_data[['user_id', 'item_id', 'context_id', new_context_column]]
            all_data_dict['test_data'] = test_data[['user_id', 'item_id', 'context_id', new_context_column]]
            # all_data_dict['positive_dict'] = df_to_positive_dict(all_data_dict['train_data'])
            try:
                t1 = time()
                all_data_dict['positive_dict'] = load_dict_from_file(self.path + '/user_pos_dict.txt')
                print('already load user positive dict', time() - t1)
            except Exception:
                all_data_dict['positive_dict'] = df_to_positive_dict(all_data_dict['train_data'])
                save_dict_to_file(all_data_dict['positive_dict'], self.path + '/user_pos_dict.txt')
            side_info['side_info_stats'] = side_info_stats
        # print("#####################  : \n ", 'UEG Train data shape : ', all_data_dict['train_data'].shape, '\nKGIN Train data shape : ', train_cf.shape)
        # print("#####################  : \n ", 'UEG test data shape : ', all_data_dict['test_data'].shape,
        #       '\nKGIN test data shape : ', test_cf.shape)
        num_ratings = len(train_data["user_id"]) + len(test_data["user_id"])
        self.logger.info("\"num_users\": %d,\"num_items\":%d,\"num_valid_items\":%d, \"num_ratings\":%d"%(num_users, num_items, num_valid_items, num_ratings))
        return [train_matrix, test_matrix, all_data_dict, side_info, num_items], [train_cf, test_cf, user_dict, n_params, graph,
                                                                                  [adj_mat_list, norm_mat_list, mean_mat_list]]

class Dataset(object):
    def __init__(self, conf, logger):
        """
        Constructor
        """
        self.logger = logger
        self.separator = conf.data_separator

        self.dataset_name = conf.dataset
        self.dataset_folder = conf.data_path
        
        data_splitter = GivenData(self.dataset_name, self.dataset_folder, conf.data_format, self.separator, self.logger)

        UEG_data, KG_data = data_splitter.load_data()
        self.train_matrix, self.test_matrix, self.all_data_dict, self.side_info, self.num_items = UEG_data
        self.train_cf, self.test_cf, self.user_dict, self.n_params, self.graph, self.mat_list = KG_data
        self.adj_mat_list, self.norm_mat_list, self.mean_mat_list = self.mat_list
        # self.test_context_list = self.all_data_dict['test_data']['context_id'].tolist() if self.side_info is not None else None
        if self.side_info is None:
            self.test_context_dict = None
        else:
            self.test_context_dict = {}
            for user, context in zip(self.all_data_dict['test_data']['user_id'].tolist(), self.all_data_dict['test_data']['context_id'].tolist()):
                self.test_context_dict[user] = context

        self.num_users, self.num_valid_items = self.train_matrix.shape
        if self.side_info is not None:
            self.num_user_features = self.side_info['side_info_stats']['num_user_features']
            self.num_item_featuers = self.side_info['side_info_stats']['num_item_features']
            self.num_context_features = self.side_info['side_info_stats']['num_context_features']
        self.logger.info('Data Loading is Done!')