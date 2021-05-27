import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()
        
        self.train_n, self.n_features = self.X_train.size()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # допишите ваш код здесь
        
        n_features_train = self.X_train.size()[1]
        self.n_objects_train = self.X_train.size()[0]
        self.n_objects_test = self.X_test.size()[0]
        
        self.feature_idx = list(range(n_features_train))
        self.object_ids = list(range(self.n_objects_train))
        self.features_fraq = int(n_features_train * colsample_bytree)
        self.object_fraq =  int(self.n_objects_train * subsample)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.X_train, self.ys_train, self.X_test, self.ys_test = map(torch.FloatTensor,
                                                                    [X_train, y_train, X_test, y_test])
        self.ys_train = torch.unsqueeze(self.ys_train, 1)
        self.ys_test = torch.unsqueeze(self.ys_test, 1)
        self.query_ids_train_u = np.unique(self.query_ids_train)
        self.query_ids_test_u = np.unique(self.query_ids_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for index in np.unique(inp_query_ids):
            current_indexes = inp_query_ids == index
            inp_feat_array[current_indexes] = StandardScaler().fit_transform(inp_feat_array[current_indexes])
            
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь
        
        lambdas = torch.zeros(self.n_objects_train, 1)
        for index in self.query_ids_train_u:
            current_index = self.query_ids_train == index
            lambda_current = self._compute_lambdas(self.ys_train[current_index], train_preds[current_index])[-1]
            lambdas[current_index] = lambda_current
        
        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                    random_state=cur_tree_idx)
        
        current_features = np.random.choice(self.feature_idx, self.features_fraq, replace=False)
        current_indexes = np.random.choice(self.object_ids, self.object_fraq, replace=False)
        
        tree.fit(self.X_train[current_indexes, :][:, current_features], -lambdas[current_indexes].numpy())
        
        return tree, current_features

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        ndcgs = []
#         print(preds)
        for index in np.unique(queries_list):
            current_index = queries_list == index
            ndcg = self._ndcg_k(true_labels[current_index].squeeze(),
                                preds[current_index].squeeze(), self.ndcg_top_k)
            if np.isnan(ndcg):
                ndcg = 0
            ndcgs.append(ndcg)
#             print(ndcgs)
        return np.mean(ndcgs)

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        
        self.best_ndcg = -1
        self.best_idx = -1

        self.trees = []
        self.tree_features = []
        train_preds = torch.zeros(self.n_objects_train, 1)
        test_preds = torch.zeros(self.n_objects_test, 1)
        
        train_ndcgs = []
        test_ndcgs = []
        
        for cur_tree_idx in tqdm(range(self.n_estimators)):
            tree, tree_features = self._train_one_tree(cur_tree_idx, train_preds)
                        
            current_X_train = self.X_train[:, tree_features]
            train_preds += self.lr * torch.FloatTensor(tree.predict(current_X_train)).reshape(-1, 1)
            current_train_ndcg = self._calc_data_ndcg(self.query_ids_train, self.ys_train, train_preds)
                        
            current_X_test = self.X_test[:, tree_features]
            test_preds += self.lr * torch.FloatTensor(tree.predict(current_X_test)).reshape(-1, 1)
            current_test_ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds)
            
            if self.best_ndcg < current_test_ndcg:
                self.best_ndcg = current_test_ndcg
                self.best_idx = cur_tree_idx + 1
                        
            train_ndcgs.append(current_train_ndcg)
            test_ndcgs.append(current_test_ndcg)
            self.trees.append(tree)
            self.tree_features.append(tree_features)
            
        
        self.trees = self.trees[:self.best_idx]
        self.tree_features = self.tree_features[:self.best_idx]
                        
        pass

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.zeros(data.shape[0], 1)
        for i in range(self.best_idx):
            current_X = data[:, self.tree_features[i]]
            preds += self.lr * torch.FloatTensor(self.trees[i].predict(current_X)).reshape(-1, 1)
        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        ideal_dcg = self._dcg_k(y_true.squeeze(), y_true.squeeze(), -1)
        if ideal_dcg:
            N = 1 / ideal_dcg
        else:
            N = 0.0
#         print(ideal_dcg)

        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true)
            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

        return Sij, gain_diff, decay_diff, delta_ndcg, lambda_update
    
    def _compute_labels_in_batch(self, y_true):
    
        # разница релевантностей каждого с каждым объектом
        rel_diff = y_true - y_true.t()

        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij

    def _compute_gain_diff(self, y_true, gain_scheme: str = 'exp2'):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff
    
    

    def compute_gain(self, y_value: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == "const":
            return y_value
        elif gain_scheme == "exp2":
            return 2 ** y_value - 1


    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        sorted_true = ys_true[indices].numpy()
        if k != -1:
            sorted_true = sorted_true[:k]
        gain = self.compute_gain(sorted_true)
        discount = [math.log2(float(x) + 1) for x in range(1, len(sorted_true) + 1)]
        discounted_gain = float((gain / discount).sum())
        return discounted_gain


    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # допишите ваш код здесь
        current_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        if ideal_dcg != 0:
            ndcg = current_dcg / ideal_dcg
        else:
            ndcg = 0
        return ndcg

    def save_model(self, path: str):
        model_info = {"trees" : self.trees,
                      "tree_features" : self.tree_features,
                     "best_ndcg": self.best_ndcg,
                     "best_idx": self.best_idx,
                     "lr": self.lr}
        with open(path, "wb") as output_file:
            pickle.dump(model_info, output_file)

    def load_model(self, path: str):
        with open(path, "rb") as input_file:
            model_info = pickle.load(input_file)
        self.trees = model_info["trees"]
        self.tree_features = model_info["tree_features"]
        self.best_ndcg = model_info["best_ndcg"]
        self.best_idx = model_info["best_idx"]
        self.lr = model_info["lr"]
