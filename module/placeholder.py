# -*- coding: utf-8 -*-

"""
输入接口文件
"""

import tensorflow as tf
from module import super_params


class ControlPlaceholder(object):
    def __init__(self):
        with tf.variable_scope(name_or_scope="control_placeholder"):
            self.global_step = tf.get_variable(name="global_step", shape=[], dtype=tf.int64,
                                               initializer=tf.zeros_initializer(), trainable=False)


class RecordPlaceholder(object):
    """
    游戏历史记录输入点位符
    """
    seq_len = super_params.seq_len

    def __init__(self):
        with tf.name_scope(name="record_placeholder"):
            # 主牌打牌游戏记录
            self.record_mcard_action = tf.placeholder(name="mcard_action",
                                                      shape=[None, self.seq_len],
                                                      dtype=tf.int64)

            # 玩家打牌的前两次记录
            self.mcard_action_2s_before = tf.placeholder(name="mcard_action_2s_before",
                                                         shape=[None, 2],
                                                         dtype=tf.int64)

            # 带牌类型游戏记录
            self.record_ktype_action = tf.placeholder(name="ktype_action",
                                                      shape=[None, self.seq_len],
                                                      dtype=tf.int64)
            # 带牌长度游戏记录
            self.record_klen_action = tf.placeholder(name="klen_action",
                                                     shape=[None, self.seq_len],
                                                     dtype=tf.int64)

            # 主牌游戏记录真实长度
            # 因为不足时会有填充
            self.true_seq_len = tf.cast(x=tf.reduce_sum(input_tensor=tf.sign(self.record_mcard_action),
                                                        axis=1), dtype=tf.int32, name="true_seq_len")


class NumDeckPlaceholder(object):
    """
    手牌数量输入占位符
    """

    def __init__(self):
        with tf.name_scope(name="num_deck_placeholder"):
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
            # 未知牌数量
            self.num_hidden_cards = tf.placeholder(dtype=tf.int64,
                                                   shape=[None],
                                                   name="num_hidden_cards")
            # 己出牌数量
            self.num_output_cards = tf.placeholder(dtype=tf.int64,
                                                   shape=[None],
                                                   name="num_output_cards")


class DeckMcardActionPlaceholder(object):
    """
    手牌可执行动作输入点位符
    """

    num_mcard_action = super_params.num_mcard_action

    def __init__(self):
        with tf.name_scope("deck_mcard_action_placeholder"):
            # 当前玩家的全部可执行主牌动作
            self.hands_action = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.num_mcard_action],
                                               name="hands_action")

            # 当前玩家的全部的主牌动作掩模
            self.inf_hands_action = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.num_mcard_action],
                                                   name="inf_hands_action")

            # 未知牌的可能动作
            self.hidden_cards_action = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, self.num_mcard_action],
                                                      name="hidden_cards_action")

            # 未知牌的可执行动作掩摸
            self.inf_hidden_cards_action = tf.placeholder(dtype=tf.float32,
                                                          shape=[None, self.num_mcard_action],
                                                          name="inf_hidden_cards_action")


class DeckVecPlaceholder(object):
    """
    手牌向量点位符
    """
    card_vec_dim = super_params.card_vec_dim

    def __init__(self):
        with tf.name_scope(name="deck_vec_placeholder"):
            # 手牌的牌组向量
            self.hands_vec = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.card_vec_dim],
                                            name="hands_vec")

            # 已经打出牌的牌组向量
            self.output_cards_vec = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.card_vec_dim],
                                                   name="output_card_vec")

            # 当前未知牌的牌组向量
            self.hidden_cards_vec = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.card_vec_dim])


class TrueLabelPlaceholder(object):
    card_vec_dim = super_params.card_vec_dim

    def __init__(self):
        with tf.name_scope("true_label"):
            self.next_mcard_action = tf.placeholder(name="next_mcard_action",
                                                    dtype=tf.int64,
                                                    shape=[None])
            # 上家手牌向量
            self.up_hands_vec = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.card_vec_dim])

            # 下家手牌向量
            self.down_hands_vec = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self.card_vec_dim])
