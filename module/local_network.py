# -*- coding: utf-8 -*-

"""
局部网络接口文件
"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from module import super_params
from module.placeholder import RecordPlaceholder
from module.placeholder import NumDeckPlaceholder
from module.placeholder import DeckVecPlaceholder
from module.placeholder import DeckMcardActionPlaceholder


class PlayCardActionEmbed(object):
    """
    打牌动作嵌入层
    """

    def __init__(self, num_action, embed_dim, name):
        self.num_action = num_action
        self.embedding_dim = embed_dim
        self.name = name

        with tf.variable_scope(name_or_scope=self.name):
            # 动作查询字典
            self.action_dict = tf.get_variable(name="action_dict",
                                               dtype=tf.float32,
                                               shape=[self.num_action, self.embedding_dim],
                                               trainable=True,
                                               initializer=initializers.xavier_initializer())

    def lookup(self, inputs):
        """
        查询字典,将索引转换为序列
        :param inputs: 动作索引序列
        :return: 转换完成的序列
        """
        inputs_seq_vec = tf.nn.embedding_lookup(params=self.action_dict, ids=inputs)

        return inputs_seq_vec


class RecordNetwork(object):
    """
    历史记录处理网络
    """

    # dropout 比率
    dropout_keep_prob = super_params.dropout_keep_prob

    # 记录主牌动作数量
    num_record_mcard_action = super_params.num_record_mcard_action
    # 记录带牌类型动作数量
    num_record_ktype_action = super_params.num_record_ktype_action
    # 记录的带牌长度数量
    num_record_klen_action = super_params.num_record_klen_action

    # 记录主牌动作嵌入维度
    record_mcard_embed_dim = super_params.record_mcard_action_embed_dim
    # 记录带牌类型嵌入维度
    record_ktype_embed_dim = super_params.record_ktype_action_embed_dim
    # 记录带牌数量嵌入维度
    record_klen_action_embed_dim = super_params.record_klen_action_embed_dim

    # LSTM 网络嵌入层数
    num_layers_of_lstm = super_params.num_layers_of_lstm
    # LSTM 网络输出维度
    lstm_output_dim = super_params.lstm_output_dim

    def __init__(self, record_placeholder):
        assert isinstance(record_placeholder, RecordPlaceholder)

        self.record_placeholder = record_placeholder

        # 主牌动作嵌入类
        self.record_mcard_action_embed = PlayCardActionEmbed(num_action=self.num_record_mcard_action,
                                                             embed_dim=self.record_mcard_embed_dim,
                                                             name="record_mcard_action_embed")
        # 带牌类型嵌入类
        self.record_ktype_action_embed = PlayCardActionEmbed(num_action=self.num_record_ktype_action,
                                                             embed_dim=self.record_ktype_embed_dim,
                                                             name="record_ktype_action_embed")
        # 带牌长度嵌入类
        self.record_klen_actiom_embed = PlayCardActionEmbed(num_action=self.num_record_klen_action,
                                                            embed_dim=self.record_klen_action_embed_dim,
                                                            name="record_klen_actiom_embed")

    def lstm(self):
        """
        LSTM模型,根据输入序列输出最后模型状态
        :param inputs: 输入的动作序列
        :return:
        """

        """
        对三个查询向量进行拼接
        """

        # 记录中动作查询
        record_action_lookup = [self.record_mcard_action_embed.lookup(self.record_placeholder.record_mcard_action),
                                self.record_ktype_action_embed.lookup(self.record_placeholder.record_ktype_action),
                                self.record_klen_actiom_embed.lookup(self.record_placeholder.record_klen_action)]

        # 对查询到的动作进行拼接
        record_action = tf.concat(record_action_lookup, axis=-1)

        # dropout
        record_action_with_drop_out = tf.nn.dropout(record_action, keep_prob=self.dropout_keep_prob)

        """
        定义LSTM网络
        """
        cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_output_dim,
                                         use_peepholes=True,
                                         state_is_tuple=True,
                                         initializer=initializers.xavier_initializer()) for _ in
                 range(self.num_layers_of_lstm)]
        muti_lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        """
        神经网络输出
        """
        outputs, states = tf.nn.dynamic_rnn(cell=muti_lstm_cell, inputs=record_action_with_drop_out,
                                            sequence_length=self.record_placeholder.true_seq_len,
                                            dtype=tf.float32)
        final_outputs = outputs[:, -1, :]

        return final_outputs

    def mcard_2s_before_lookup(self):
        """
        对打牌之前的再次动作进行向量查询
        :return:
        """
        mcard_2s_before_vec = self.record_mcard_action_embed.lookup(self.record_placeholder.mcard_action_2s_before)
        mcard_2s_before_vec_flatten = tf.reshape(mcard_2s_before_vec, [-1, 2 * self.record_mcard_embed_dim])
        mcard_2s_before_vec_flatten_with_dropout = tf.nn.dropout(mcard_2s_before_vec_flatten,
                                                                 keep_prob=self.dropout_keep_prob)
        return mcard_2s_before_vec_flatten_with_dropout

    def output(self):
        """
        模型输出
        :return:
        """
        return self.lstm(), self.mcard_2s_before_lookup()


class NumDeckEmbedNetwork(object):
    """
    将卡牌数量转换为向量
    将玩家手牌数量,未知手牌数量,己出手牌数量,上家手牌数量,下家手牌数量进行向量查询,并拼接
    """
    # 几个参数
    dropout_keep_prob = super_params.dropout_keep_prob
    num_deck = super_params.num_deck
    num_deck_encode_dim = super_params.num_deck_encode_dim

    def __init__(self, num_deck_placeholder):
        assert isinstance(num_deck_placeholder, NumDeckPlaceholder)
        self.num_deck_placeholder = num_deck_placeholder
        with tf.variable_scope(name_or_scope="num_deck_embed"):
            self.num_deck_dict = tf.get_variable(name="num_deck_dict",
                                                 dtype=tf.float32,
                                                 shape=[self.num_deck, self.num_deck_encode_dim],
                                                 trainable=True,
                                                 initializer=initializers.xavier_initializer())

    def lookup(self):
        """
        向量查询
        :return:
        """

        num_hands_vec = tf.nn.embedding_lookup(params=self.num_deck_dict, ids=self.num_deck_placeholder.num_hands)
        num_hidden_card_vec = tf.nn.embedding_lookup(params=self.num_deck_dict,
                                                     ids=self.num_deck_placeholder.num_hidden_cards)
        num_output_card_vec = tf.nn.embedding_lookup(params=self.num_deck_dict,
                                                     ids=self.num_deck_placeholder.num_output_cards)
        num_up_hands_vec = tf.nn.embedding_lookup(params=self.num_deck_dict,
                                                  ids=self.num_deck_placeholder.num_up_hands)
        num_down_hands_vec = tf.nn.embedding_lookup(params=self.num_deck_dict,
                                                    ids=self.num_deck_placeholder.num_down_hands)

        """
        向量拼接
        """
        num_deck_vec = tf.concat(values=[num_hands_vec,
                                         num_hidden_card_vec,
                                         num_output_card_vec,
                                         num_up_hands_vec,
                                         num_down_hands_vec], axis=-1)
        """
        dropout
        """
        num_deck_vec_with_dropout = tf.nn.dropout(num_deck_vec, keep_prob=self.dropout_keep_prob)

        return num_deck_vec_with_dropout

    def output(self):
        return self.lookup()


class DeckMcardActionEncodeNetwork(object):
    """
    对手牌可执行的主牌动作向量进行压缩
    """

    # 几个参数
    dropout_keep_prob = super_params.dropout_keep_prob
    num_mcard_action = super_params.num_mcard_action
    deck_mcard_action_encode_dim = super_params.deck_mcard_action_encode_dim

    def __init__(self, deck_mcard_action_placeholder):
        assert isinstance(deck_mcard_action_placeholder, DeckMcardActionPlaceholder)
        self.deck_mcard_action_placeholder = deck_mcard_action_placeholder
        """
        压缩网络参数
        """
        with tf.variable_scope(name_or_scope="deck_mcard_action_encode"):
            self.weights = tf.get_variable(name="weights",
                                           shape=[self.num_mcard_action, self.deck_mcard_action_encode_dim],
                                           trainable=True,
                                           initializer=initializers.xavier_initializer())
            self.bias = tf.get_variable(name="bias",
                                        shape=[self.deck_mcard_action_encode_dim],
                                        trainable=True,
                                        initializer=tf.zeros_initializer())

    def encode(self):
        """
        数据压缩
        :return:
        """
        """
        压缩
        """
        hands_action_encode = tf.nn.relu(self.deck_mcard_action_placeholder.hands_action @ self.weights + self.bias)
        hidden_card_action_encode = tf.nn.relu(
            self.deck_mcard_action_placeholder.hidden_cards_action @ self.weights + self.bias)

        """
        拼接
        """
        deck_mcard_action_encode = tf.concat(values=[hands_action_encode, hidden_card_action_encode], axis=-1)

        """
        dropout
        """
        deck_mcard_action_encode_with_dropout = tf.nn.dropout(deck_mcard_action_encode,
                                                              keep_prob=self.dropout_keep_prob)
        return deck_mcard_action_encode_with_dropout

    def output(self):
        return self.encode()


class DeckVecEncodeNetwork(object):
    """
    对手牌向量进行信息压缩
    """
    dropout_keep_prob = super_params.dropout_keep_prob
    card_vec_dim = super_params.card_vec_dim
    card_vec_encode_dim = super_params.card_vec_encode_dim

    def __init__(self, deck_vec_placeholder):
        assert isinstance(deck_vec_placeholder, DeckVecPlaceholder)
        self.deck_vec_placeholder = deck_vec_placeholder

        """
        压缩网络参数
        """
        with tf.variable_scope(name_or_scope="deck_vec_encode", dtype=tf.float32):
            self.weights = tf.get_variable(name="weights",
                                           shape=[self.card_vec_dim, self.card_vec_encode_dim],
                                           trainable=True,
                                           initializer=initializers.xavier_initializer())
            self.bias = tf.get_variable(name="bias",
                                        shape=[self.card_vec_encode_dim],
                                        trainable=True,
                                        initializer=tf.zeros_initializer())

    def encode(self):
        """
        数据压缩
        :return:
        """

        """
        压缩
        """
        hands_vec_encode = tf.nn.relu(self.deck_vec_placeholder.hands_vec @ self.weights + self.bias)
        output_card_vec_encode = tf.nn.relu(self.deck_vec_placeholder.output_cards_vec @ self.weights + self.bias)
        hidden_card_vec_encode = tf.nn.relu(self.deck_vec_placeholder.hidden_cards_vec @ self.weights + self.bias)

        """
        拼接
        """
        deck_vec_encode = tf.concat(values=[hands_vec_encode,
                                            output_card_vec_encode,
                                            hidden_card_vec_encode], axis=-1)
        """
        dropout
        """
        deck_vec_encode_with_dropout = tf.nn.dropout(deck_vec_encode, keep_prob=self.dropout_keep_prob)

        return deck_vec_encode_with_dropout

    def output(self):
        return self.encode()


if __name__ == "__main__":
    a = RecordPlaceholder()
    RecordNetwork(a)
    d = McardAction2sBeforePlaceholder()
    McardAction2sBeforeNetowk(d)
