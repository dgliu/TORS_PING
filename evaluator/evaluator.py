from utility.Tool import csr_to_user_dict, typeassert, pad_sequences, randint_choice, argmax_top_k
from scipy.sparse import csr_matrix
from utility.DataIterator import DataIterator
import numpy as np
from evaluator.backend import eval_score_matrix_loo
import pandas as pd


def recall_at_k(r, all_pos_num):
    r = np.asfarray(r)
    return np.sum(r) / all_pos_num


def dcg_at_k(r, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    sent_list = [1.0]*len(ground_truth)
    dcg_max = dcg_at_k(sent_list, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, method) / dcg_max


class AbstractEvaluator(object):
    """Basic class for evaluator.
    """
    def __init__(self):
        pass

    def metrics_info(self):
        raise NotImplementedError

    def evaluate(self, ranking_score):
        raise NotImplementedError


class LeaveOneOutEvaluator(AbstractEvaluator):
    """Evaluator for leave one out ranking task.
    """
    # @typeassert(train_matrix=csr_matrix, test_matrix=csr_matrix, top_k=(int, list, tuple))
    def __init__(self, dataset, train_matrix, test_matrix,num_valid_items, test_context_dict=None, top_k=20):
        super(LeaveOneOutEvaluator, self).__init__()
        # use the whole test items
        self.dataset = dataset
        self.max_top = top_k if isinstance(top_k, int) else max(top_k)
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k) + 1
        else:
            self.top_show = np.sort(top_k)
        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)
        user_maxid = max(self.user_pos_test.keys())
        self.user_pos_test_arr = np.array([self.user_pos_test[x] if x in self.user_pos_test else 0 for x in  range(user_maxid+1)])
        self.user_rating_mask = np.ones((user_maxid+1,num_valid_items))
        for user_id in self.user_pos_train.keys():
            self.user_rating_mask[user_id][self.user_pos_train[user_id]] = -np.inf

        self.test_context_dict = test_context_dict
        self.metrics_num = 2

    def metrics_info(self):
        HR = '\t'.join([("Recall@"+str(k)).ljust(12) for k in self.top_show])
        NDCG = '\t'.join([("NDCG@" + str(k)).ljust(12) for k in self.top_show])
        MRR = '\t'.join([("MRR@" + str(k)).ljust(12) for k in self.top_show])
        metric = '\t'.join([HR, NDCG])
        # return metric
        return "metrics:\t%s" % metric

    def auc(self, vector_true_dense, vector_predict, **unused):
        pos_indexes = np.where(vector_true_dense == 1)[0]
        sort_indexes = np.argsort(vector_predict)
        rank = np.nonzero(np.in1d(sort_indexes, pos_indexes))[0] + 1
        return (
                       np.sum(rank) - len(pos_indexes) * (len(pos_indexes) + 1) / 2
               ) / (
                       len(pos_indexes) * (len(vector_predict) - len(pos_indexes))
               )

    def evaluate(self, model):
        # B: batch size
        # N: the number of items
        test_batch_size = model.test_batch_size

        test_users = DataIterator(list(self.user_pos_test.keys()), batch_size=test_batch_size, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            test_items = []
            for user in batch_users:
                num_item = len(self.user_pos_test[user])
                if num_item != 1:
                    raise ValueError("the number of test item of user %d is %d" % (user, num_item))
                test_items.append(self.user_pos_test[user][0])
            ranking_score = model.predict(batch_users, None)  # (B,N)
            ranking_score = np.array(ranking_score)

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(batch_users):
                train_items = self.user_pos_train[user]
                ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix_loo(ranking_score, test_items, top_k=self.max_top, thread_num=None)  # (B,k*metric_num)
            batch_result.append(result)

        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)
        final_result = np.mean(all_user_result, axis=0)  # mean

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])
        final_result = final_result[:, self.top_show - 1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in final_result])
        return final_result, buf

    def evaluate4recall(self, model, num_recall, recall_type='MF'):
        # B: batch size
        # N: the number of items
        test_users = DataIterator(list(self.user_pos_test.keys()), batch_size=2048, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            result = []
            if recall_type == 'MF':
                ranking_score = model.predict(batch_users, None)  # (B,N)
                ranking_score = np.array(ranking_score)

                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                for idx, user in enumerate(batch_users):
                    ranking_score_cur_user = ranking_score[idx]
                    all_pos_items = self.user_pos_train[user]
                    all_pos_items.extend(self.user_pos_test[user])
                    ranking_score_cur_user[all_pos_items] = -np.inf     # mark scores of all positive items as -inf
                    recall_item_cur = np.argsort(-ranking_score_cur_user)   # sort in decent order, return id
                    recall_item_cur = recall_item_cur[0: num_recall - 1]    # only keep first (num_recall - 1) items
                    # recall_item_cur = np.append(recall_item_cur, self.user_pos_test[user])    # include test item
                    recall_item_cur = np.sort(recall_item_cur)
                    result.append(recall_item_cur)
            else:
                for idx, user in enumerate(batch_users):
                    all_pos_items = self.user_pos_train[user]
                    all_pos_items.extend(self.user_pos_test[user])
                    recall_item_cur = randint_choice(model.num_valid_items, num_recall - 1, replace=False, exclusion=all_pos_items)
                    recall_item_cur = np.sort(recall_item_cur)
                    result.append(recall_item_cur)

            batch_result.append(result)

        # concatenate the batch results to a matrix
        final_result = np.concatenate(batch_result, axis=0)
        return final_result

    def evaluate4CARS(self, model):
        # B: batch size
        # N: the number of items
        test_batch_size = model.test_batch_size

        user_test_list = self.dataset.all_data_dict['test_data'].user_id.tolist()
        item_test_list = self.dataset.all_data_dict['test_data'].item_id.tolist()
        context_no_id_test_list = self.dataset.all_data_dict['test_data'][self.dataset.all_data_dict['test_data'].columns[-1]].tolist()
        test_context_list = [self.test_context_dict[u] for u in user_test_list]
        test_iter = DataIterator(user_test_list, item_test_list, test_context_list, context_no_id_test_list, batch_size=test_batch_size, shuffle=False,
                                 drop_last=False)
        batch_result = []
        user_list = []
        for batch_users, batch_items, batch_contexts, batch_contexts_no_id in test_iter:
            test_items = batch_items
            ranking_score = model.predict(batch_users, batch_contexts)  # (B,N)
            ranking_score = np.array(ranking_score)

            for idx, user in enumerate(batch_users):
                train_items = self.user_pos_train[user]
                ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix_loo(ranking_score, test_items, top_k=self.max_top,
                                           thread_num=None)  # (B,k*metric_num)

            batch_result.extend(result.tolist())
            user_list.extend(batch_users)

        # concatenate the batch results to a matrix
        user2result_df = pd.DataFrame()
        user2result_df['user'] = user_list
        user2result_df['result'] = batch_result
        groups = user2result_df.groupby('user')
        final_result = []
        for group in groups:
            result_one_user = np.array(group[1]['result'].tolist())
            result_recall = result_one_user[:, self.max_top-1]  # [I, 1]
            result_topk = result_one_user[:, self.max_top*2-1]
            ndcg_one_user = np.mean(result_topk)
            recall_one_user = recall_at_k(result_recall, result_recall.shape[0])
            final_result.append([recall_one_user, ndcg_one_user])
        all_user_result = np.array(final_result)
        final_result = np.mean(all_user_result, axis=0)  # mean

        buf = '\t'.join([("%.4f" % x).ljust(12) for x in final_result])

        return final_result, buf

