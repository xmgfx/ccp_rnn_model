# -*- coding: utf-8 -*-
"""
模型超参数
"""

"""
迭代次数
"""
num_epoch = 10

"""
批次大小
"""
batch_size = 16

"""
dropout 保留比例
"""
dropout_keep_prob = 0.5
"""
正则化权重
"""
l2_regularizer_scale = 1e-3

"""
学习率
"""
learning_rate_base = 0.1
learning_rate_decay = 0.99999999
learning_rate_decay_step = 100

learning_rate_standard = 1e-3

"""
序列长度
"""
seq_len = 100
"""
主牌动作
"""
# 主牌动作数
num_mcard_action = 309
# 嵌入维度
deck_mcard_action_encode_dim = 100

"""
记录中的主牌动作
因为加入了START SEP 和 None 所以会大一点
"""
num_record_mcard_action = num_mcard_action + 3
record_mcard_action_embed_dim = 100
"""
带牌类型动作数量
"""
num_record_ktype_action = 5
record_ktype_action_embed_dim = 2
"""
带牌长度动作数量
"""
num_record_klen_action = 8
record_klen_action_embed_dim = 4

"""
手牌向量
"""
card_vec_dim = 69
card_vec_encode_dim = 25

"""
手牌数
"""
# 最大手牌数
num_deck = 54
# 手牌嵌入维度
num_deck_encode_dim = 27

"""
LSTM网络
"""
num_layers_of_lstm = 2
lstm_output_dim = 100
mcard_action_embed_dim = 100
