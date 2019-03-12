# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin

import re
import string

from config import *

# 读取文件数据
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
subm = pd.read_csv(sample_submission_file)
test_labels = pd.read_csv(test_labels_file)

# 增加列
train['none'] = 1-train[label_cols].max(axis=1)

# 填充缺失的评论
COMMENT = "comment_text"
train[COMMENT].fillna('UNKNOWN', inplace=True)
test[COMMENT].fillna('UNKNOWN', inplace=True)

# 正则化定义
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).strip()

# NB-SVM 分类器
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C = 1.0, dual = False, n_jobs = 1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def get_mdl(self, train_x, y):
        check_X_y(train_x, y, accept_sparse=True)
        y = y.values
        # 朴素贝叶斯特征方程
        def pri(x, y_i, y):
            p = x[y_i == y].sum(0)
            return (p+1)/((y_i == y).sum() + 1)

        self._r = sparse.csc_matrix(np.log(pri(train_x, 1, y)/pri(train_x, 0, y)))
        m = LogisticRegression(C = self.C, dual = self.dual)
        nb_x = train_x.multiply(self._r)
        self._clf = m.fit(nb_x, y)
        return self

    def predict_prob(self, test_x):
        # check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(test_x.multiply(self._r))[:, 1]

if __name__ == "__main__":
    vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize, \
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1, \
                          smooth_idf=1, sublinear_tf=1)
    train_term_doc = vec.fit_transform(train[COMMENT])
    test_term_doc = vec.transform(test[COMMENT])

    train_x = train_term_doc
    test_x = test_term_doc

    # 分类预测，每次针对一类标签
    preds = np.zeros((len(test), len(label_cols)))
    for i, j in enumerate(label_cols):
        model = NbSvmClassifier(C=4, dual=True).get_mdl(train_x, train[j])
        preds[:, i] = model.predict_prob(test_x)

    # 预测结果写入cvs文件
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
    submission.to_csv(submission_file, index=False)

    # 预测结果处理显示
    acc_list = []
    for i, j in enumerate(label_cols):
        accuracy = (abs(submission[j].values - test_labels[j].values) < 0.5).sum()/((test_labels[j].values!=-1).sum())
        acc_list.append(accuracy)
        print("predict accuracy of {}: {}".format(j, accuracy))
    print("avg predict accuracy: ".format(sum(acc_list)/len(acc_list)))
