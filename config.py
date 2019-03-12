# -*- coding:utf8 -*-
# 文件路径
data_dir = './data'
train_file = data_dir + '/train.csv'
test_file = data_dir + '/test.csv'
test_labels_file = data_dir + '/test_labels.csv'
sample_submission_file = data_dir + '/sample_submission.csv'
submission_file = data_dir + '/submission.csv'

# 标签列名称
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
