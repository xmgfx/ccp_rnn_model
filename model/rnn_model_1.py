# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from data import loader
import numpy as np


class SuppeParams(object):
    """
    定义超参数
    """

    # 三个初化器
    _xavier_initializer = initializers.xavier_initializer()
    _he_initializer = tf.contrib.layers.variance_scaling_initializer()
    _zeros_initializer = tf.zeros_initializer()

    def __init__(self):
        with tf.variable_scope(name_or_scope="super_params"):
            # 全局步数
            self.global_step = tf.get_variable(name="global_step",
                                               shape=(),
                                               dtype=tf.int64)


class RecordPlaceholder(object):
    """
    游戏历史记录输入点位符
    """

    # 默认序列长度
    seq_len = 100

    def __init__(self):
        with tf.name_scope(name="game_record_placeholder"):
            # 主牌打牌游戏记录
            self.mcard_action_record = tf.placeholder(name="mcard_action",
                                                      shape=[None, self.seq_len],
                                                      dtype=tf.int64)

            # 带牌类型游戏记录
            self.ktype_action_record = tf.placeholder(name="ktype_action",
                                                      shape=[None, self.seq_len],
                                                      dtype=tf.int64)
            # 带牌长度游戏记录
            self.klen_action_record = tf.placeholder(name="klen_action",
                                                     shape=[None, self.seq_len],
                                                     dtype=tf.int64)

            # 主牌游戏记录真实长度
            # 因为不足时会有填充
            self.record_true_len = tf.cast(x=tf.reduce_sum(input_tensor=tf.sign(self.mcard_action_record),
                                                           axis=1), dtype=tf.int32, name="true_len")


class ControlPlaceholder(object):
    """
    模型的控制占位符
    """

    def __init__(self):
        with tf.name_scope(name="control_placeholder"):
            # 是否是训练状态
            self.is_train = tf.placeholder(name="is_train",
                                           dtype=tf.bool,
                                           shape=[])
            # drop_out保留比例
            self.dropout_keep_prob = tf.placeholder(name="dropout_keep_prob",
                                                    dtype=tf.float32,
                                                    shape=[])
            # 学习率
            self.learning_rate = tf.placeholder(name="learning_rate",
                                                dtype=tf.float32,
                                                shape=[])


class CurrentStatePlaceholder(object):
    """
    当前游戏状态输入占位符
    """

    # 主牌动作数
    num_mcard_action = 309
    # 牌组向量维度
    card_vec_dim = 69

    def __init__(self):
        with tf.name_scope(name="current_state_placeholder"):
            # 已经打出牌的牌组向量
            self.output_cards_vec = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.card_vec_dim],
                                                   name="output_card_vec")
            # 手牌的牌组向量
            self.hands_vec = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.card_vec_dim],
                                            name="hands_vec")

            self.mcard_action_2s_before = tf.placeholder(dtype=tf.float32,
                                                         shape=[None, 2],
                                                         name="mcard_action_2s_before")

            # 当前玩家的全部可执行主牌动作
            self.hands_action = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.num_mcard_action],
                                               name="hands_action")

            # 当前玩家的全部的主牌动作掩模
            self.inf_hands_action = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.num_mcard_action],
                                                   name="inf_hands_action")

            # 当前未知牌的牌组向量
            self.hidden_cards_vec = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.card_vec_dim])

            # 未知牌的可能动作
            self.hidden_cards_action = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, self.num_mcard_action],
                                                      name="hidden_cards_action")

            # 未知牌的可执行动作掩摸
            self.inf_hidden_cards_action = tf.placeholder(dtype=tf.float32,
                                                          shape=[None, self.num_mcard_action],
                                                          name="inf_hidden_cards_action")
            # 自己的手牌数量
            self.num_hands = tf.placeholder(dtype=tf.int64,
                                            shape=[None],
                                            name="num_hands")
            # 上家手牌数量
            self.num_up_hands = tf.placeholder(dtype=tf.int64,
                                               shape=[None],
                                               name="num_up_hands")
            # 下家手牌数量
            self.num_down_hands = tf.placeholder(dtype=tf.int64,
                                                 shape=[None],
                                                 name="num_down_hands")
            self.num_hidden_cards = tf.placeholder(dtype=tf.int64,
                                                   shape=[None],
                                                   name="num_hidden_cards")
            self.num_output_cards = tf.placeholder(dtype=tf.int64,
                                                   shape=[None],
                                                   name="num_output_cards")


class PredictLabelsPlaceholder(object):
    # 主牌动作数
    num_mcard_action = 309
    # 牌组向量维度
    card_vec_dim = 69

    def __init__(self):
        with tf.name_scope(name="predict_labels_placeholder"):
            # 上家可执行动作
            self.up_hands_action = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.num_mcard_action],
                                                  name="up_hands_action")

            self.next_mcard_action = tf.placeholder(dtype=tf.float32,
                                                    shape=[None])

            # 下家可执行动作
            self.down_hands_action = tf.placeholder(dtype=tf.float32,
                                                    shape=[None, self.num_mcard_action],
                                                    name="down_hands_action")



class PlayCardActionEmbedding(object):
    """
    打牌动作嵌入层
    """
    # 三个初始化器
    _xvaier_initializer = initializers.xavier_initializer()
    _he_initializer = tf.contrib.layers.variance_scaling_initializer()
    _zeros_initializer = tf.zeros_initializer()
    _normal_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)

    def __init__(self, num_action, embedding_dim, name):
        self.num_action = num_action
        self.embedding_dim = embedding_dim
        self.name = name

        with tf.variable_scope(name_or_scope=self.name):
            # 动作查询字典
            self.action_dict = tf.get_variable(name="action_dict",
                                               dtype=tf.float32,
                                               shape=[self.num_action, self.embedding_dim],
                                               trainable=True,
                                               initializer=self._normal_initializer)

    def lookup(self, action_seq):
        """
        查询字典,将索引转换为序列
        :param action_seq: 动作索引序列
        :return: 转换完成的序列
        """
        with tf.variable_scope(name_or_scope=self.name):
            action_seq_lookup = tf.nn.embedding_lookup(params=self.action_dict, ids=action_seq)

        return action_seq_lookup


class CardVec(object):
    """
    牌组向量
    """

    # 牌组向量维度
    card_vec_dim = 69

    # 三个初始化器
    _xvaier_initializer = initializers.xavier_initializer()
    _he_initializer = tf.contrib.layers.variance_scaling_initializer()
    _zeros_initializer = tf.zeros_initializer()
    _normal_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)

    def __init__(self, output_dim):
        with tf.variable_scope(name_or_scope="card_vec_encode", dtype=tf.float32):
            self.weights = tf.get_variable(name="weights",
                                           shape=[self.card_vec_dim, output_dim],
                                           trainable=True,
                                           initializer=self._normal_initializer)
            self.bias = tf.get_variable(name="bias",
                                        shape=[output_dim],
                                        trainable=True,
                                        initializer=self._zeros_initializer)

    def encode(self, inputs):
        return tf.nn.relu(inputs @ self.weights + self.bias)


class CardALLMcardActionEncode(object):
    num_mcard_action = 309
    card_all_mcard_action_embedding_dim = 100

    # 三个初始化器
    _xvaier_initializer = initializers.xavier_initializer()
    _he_initializer = tf.contrib.layers.variance_scaling_initializer()
    _zeros_initializer = tf.zeros_initializer()
    _normal_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)

    def __init__(self):
        with tf.variable_scope(name_or_scope="card_all_mcard_action_encode", dtype=tf.float32):
            self.weights = tf.get_variable(name="weights",
                                           shape=[self.num_mcard_action, self.card_all_mcard_action_embedding_dim],
                                           trainable=True,
                                           initializer=self._normal_initializer)
            self.bias = tf.get_variable(name="bias",
                                        shape=[self.card_all_mcard_action_embedding_dim],
                                        trainable=True,
                                        initializer=self._zeros_initializer)

    def encode(self, inputs):
        return tf.nn.relu(inputs @ self.weights + self.bias)


class NumCardEmbedding(object):
    max_num_card = 20
    num_card_embedding_dim = 10

    # 三个初始化器
    _xvaier_initializer = initializers.xavier_initializer()
    _he_initializer = tf.contrib.layers.variance_scaling_initializer()
    _zeros_initializer = tf.zeros_initializer()
    _normal_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)

    def __init__(self):
        with tf.variable_scope(name_or_scope="num_card_embedding", dtype=tf.float32):
            self.num_card_dict = tf.get_variable(name="num_card_dict",
                                                 shape=[self.max_num_card + 1, self.num_card_embedding_dim],
                                                 trainable=True,
                                                 initializer=self._normal_initializer)

    def lookup(self, inputs):
        num_card_vec = tf.nn.embedding_lookup(params=self.num_card_dict, ids=inputs)
        return num_card_vec


def lstm(inputs, true_seq_len):
    """
    LSTM模型,根据输入序列输出最后模型状态
    :param inputs: 输入的动作序列
    :param true_seq_len: 序列的真实长度
    :return:
    """
    # 神经元层数
    num_cell = 2
    # 输入序列维度
    num_units = 100
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=num_units,

                                     use_peepholes=True,
                                     state_is_tuple=True,
                                     initializer=tf.random_normal_initializer(stddev=1e-2)) for _ in range(num_cell)]
    muti_lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    outputs, states = tf.nn.dynamic_rnn(cell=muti_lstm_cell, inputs=inputs, sequence_length=true_seq_len,
                                        dtype=tf.float32)
    final_outputs = outputs[:, -1, :]

    return final_outputs


def rnn_model():
    rnn_file_path = "../data/sample100w"

    rnn_data_loader = loader.Loader(path=rnn_file_path)

    num_epoch = 10
    batch_size = 128

    num_mcard_action = 309
    # 打牌记录的全部动作数
    num_record_mcard_action = 312
    # 打牌记录中的带牌类型数
    num_record_ktype_action = 5
    # 打牌记录中的带牌长度类型数
    num_record_klen_action = 8

    # 打牌记录的主牌动作嵌入维度
    record_mcard_action_output_dim = 100
    # 打牌动作的带牌类型嵌入维度
    record_ktype_action_output_dim = 10
    # 打牌动作中的带牌长度嵌入维度
    record_klen_action_output_dim = 30

    # 牌组向量维度
    card_vec_encode_dim = 60

    # 超参数
    super_params = SuppeParams()

    # 控制输入口
    control_placeholder = ControlPlaceholder()

    # 游戏历史记录输入口
    game_record_placeholder = RecordPlaceholder()

    """
    主牌动作
    """
    record_mcard_action_embed = PlayCardActionEmbedding(num_action=num_record_mcard_action,
                                                        embedding_dim=record_mcard_action_output_dim,
                                                        name="record_mcard_action_embed")
    record_mcard_action_seq = record_mcard_action_embed.lookup(action_seq=game_record_placeholder.mcard_action_record)
    record_mcard_action_seq_with_dropout = tf.nn.dropout(record_mcard_action_seq,
                                                         keep_prob=control_placeholder.dropout_keep_prob)

    """
    带牌类型动作
    """
    record_ktype_action_embed = PlayCardActionEmbedding(num_action=num_record_ktype_action,
                                                        embedding_dim=record_ktype_action_output_dim,
                                                        name="record_ktype_action_embed")
    record_ktype_action_seq = record_ktype_action_embed.lookup(action_seq=game_record_placeholder.ktype_action_record)

    record_ktype_action_seq_with_dropout = tf.nn.dropout(record_ktype_action_seq,
                                                         keep_prob=control_placeholder.dropout_keep_prob)

    """
    带牌长度动作
    """
    record_klen_action_embed = PlayCardActionEmbedding(num_action=num_record_klen_action,
                                                       embedding_dim=record_klen_action_output_dim,
                                                       name="record_klen_action_embed")
    record_klen_action_seq = record_klen_action_embed.lookup(action_seq=game_record_placeholder.klen_action_record)
    record_klen_action_seq_with_dropout = tf.nn.dropout(record_klen_action_seq,
                                                        keep_prob=control_placeholder.dropout_keep_prob)

    """
    主牌动作 带牌类型动作 带牌长度动作 拼接
    """
    record_action_seq = tf.concat(values=[record_mcard_action_seq_with_dropout,
                                          record_ktype_action_seq_with_dropout,
                                          record_klen_action_seq_with_dropout], axis=-1)

    """
    神经网络预测,以最后输出作为当前打牌的局势状态
    """
    record_vec = lstm(inputs=record_action_seq, true_seq_len=game_record_placeholder.record_true_len)
    record_vec_with_dropout = tf.nn.dropout(record_vec, keep_prob=control_placeholder.dropout_keep_prob)

    current_state_placeholder = CurrentStatePlaceholder()

    card_vec = CardVec(output_dim=card_vec_encode_dim)

    """
    手牌向量压缩编码
    """
    hands_vec = card_vec.encode(inputs=current_state_placeholder.hands_vec)
    hands_vec_with_dropout = tf.nn.dropout(hands_vec,
                                           keep_prob=control_placeholder.dropout_keep_prob)

    """
    已经打出牌压缩编码
    """
    output_cards_vec = card_vec.encode(inputs=current_state_placeholder.output_cards_vec)
    output_cards_vec_with_dropout = tf.nn.dropout(output_cards_vec,
                                                  keep_prob=control_placeholder.dropout_keep_prob)

    """
    未知牌压缩编码
    """
    hidden_cards_vec = card_vec.encode(inputs=current_state_placeholder.hidden_cards_vec)
    hidden_cards_vec_with_dropout = tf.nn.dropout(hidden_cards_vec,
                                                  keep_prob=control_placeholder.dropout_keep_prob)

    num_card_embed = NumCardEmbedding()

    """
    手牌数量压缩编码
    """
    num_hands = num_card_embed.lookup(inputs=current_state_placeholder.num_hands)
    num_hands_with_dropout = tf.nn.dropout(num_hands, keep_prob=control_placeholder.dropout_keep_prob)

    """
    已经打出牌的数量压缩编码
    """
    num_output_cards = num_card_embed.lookup(inputs=current_state_placeholder.num_output_cards)
    num_output_cards_with_dropout = tf.nn.dropout(num_output_cards, keep_prob=control_placeholder.dropout_keep_prob)

    "未知牌数量压缩编码"
    num_hidden_cards = num_card_embed.lookup(inputs=current_state_placeholder.num_hidden_cards)
    num_hidden_cards_with_dropout = tf.nn.dropout(num_hidden_cards, keep_prob=control_placeholder.dropout_keep_prob)

    """
    上家牌数量压缩编码
    """
    num_up_hands = num_card_embed.lookup(inputs=current_state_placeholder.num_up_hands)
    num_up_hands_with_dropout = tf.nn.dropout(num_up_hands, keep_prob=control_placeholder.dropout_keep_prob)

    """
    下家牌数量压缩编码
    """
    num_down_hands = num_card_embed.lookup(inputs=current_state_placeholder.num_down_hands)
    num_down_hands_dropout = tf.nn.dropout(num_down_hands, keep_prob=control_placeholder.dropout_keep_prob)

    card_all_mcard_action_encode = CardALLMcardActionEncode()

    """
    当前手牌所有可执行动作压缩
    """
    hands_all_action = card_all_mcard_action_encode.encode(inputs=current_state_placeholder.hands_action)
    hands_all_action_with_dropout = tf.nn.dropout(hands_all_action, keep_prob=control_placeholder.dropout_keep_prob)

    """
    未知手牌所有可执行动作压缩
    """
    hidden_cards_all_action = card_all_mcard_action_encode.encode(inputs=current_state_placeholder.hidden_cards_action)
    hidden_cards_all_action_with_dropout = tf.nn.dropout(hidden_cards_all_action,
                                                         keep_prob=control_placeholder.dropout_keep_prob)

    total_message = tf.concat(values=[record_vec_with_dropout,
                                      hands_vec_with_dropout,
                                      hidden_cards_vec_with_dropout,
                                      output_cards_vec_with_dropout,
                                      hands_all_action_with_dropout,
                                      hidden_cards_all_action_with_dropout,
                                      num_hands_with_dropout,
                                      num_hidden_cards_with_dropout,
                                      num_output_cards_with_dropout,
                                      num_down_hands_dropout,
                                      num_up_hands_with_dropout], axis=-1)

    next_mcard_pred = tf.layers.dense(inputs=total_message, units=num_mcard_action)

    next_mcard_pred += current_state_placeholder.inf_hands_action
    predict_label_placeholder = PredictLabelsPlaceholder()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for n_epoch in range(num_epoch):
            for batch_data in rnn_data_loader.read_batch(batch_size=batch_size):
                print(batch_data.keys())
                feed_dict = {
                    control_placeholder.dropout_keep_prob: 0.5,
                    game_record_placeholder.mcard_action_record: np.array(batch_data['mcard_action_record'],
                                                                          dtype=np.int),
                    game_record_placeholder.ktype_action_record: np.array(batch_data['ktype_action_record'],
                                                                          dtype=np.int),
                    game_record_placeholder.klen_action_record: np.array(batch_data['klen_action_record'],
                                                                         dtype=np.int),
                    current_state_placeholder.mcard_action_2s_before: np.array(batch_data['mcard_record_2s_before'],
                                                                               dtype=np.int),
                    current_state_placeholder.output_cards_vec: np.array(batch_data['output_cards_vec'], dtype=np.int),
                    current_state_placeholder.hidden_cards_vec: np.array(batch_data['hidden_cards_vec'], dtype=np.int),
                    current_state_placeholder.hands_vec: np.array(batch_data['hands_vec'], dtype=np.int),
                    current_state_placeholder.hidden_cards_action: np.array(batch_data['hidden_cards_action_vec'],
                                                                            dtype=np.int),
                    current_state_placeholder.inf_hidden_cards_action: np.array(
                        batch_data['inf_hidden_cards_action_vec'], dtype=np.float),
                    current_state_placeholder.hands_action: np.array(batch_data['hands_action_vec'], dtype=np.int),
                    current_state_placeholder.inf_hands_action: np.array(batch_data['inf_hands_action_vec'],
                                                                         dtype=np.float),
                    current_state_placeholder.num_output_cards: np.array(batch_data['num_output_cards'], dtype=np.int),
                    current_state_placeholder.num_hidden_cards: np.array(batch_data['num_hidden_cards'], dtype=np.int),
                    current_state_placeholder.num_hands: np.array(batch_data['num_hands'], dtype=np.int),
                    current_state_placeholder.num_up_hands: np.array(batch_data['num_up_hands'], dtype=np.int),
                    current_state_placeholder.num_down_hands: np.array(batch_data['num_down_hands'], dtype=np.int),
                    predict_label_placeholder.next_mcard_action: np.array(batch_data['next_mcard_action_label'],
                                                                          dtype=np.int),
                    predict_label_placeholder.down_hands_vec: np.array(batch_data['down_hands_vec'], dtype=np.int),
                    predict_label_placeholder.up_hands_vec: np.array(batch_data['up_hands_vec'], dtype=np.int)
                }
                tmp = sess.run(record_mcard_action_embed, feed_dict=feed_dict)
                print(tmp)
                break
            break


if __name__ == "__main__":
    rnn_model()
