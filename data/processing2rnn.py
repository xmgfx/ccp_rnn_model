# -*- coding: utf-8 -*-

"""
RNN模型数据处理
"""

import pickle
import os
import numpy as np
import pandas as pd
import logging
from difflib import Differ
from collections import Counter
from collections import OrderedDict

# 配置调试
logging.basicConfig(level=logging.DEBUG, format="[%(threadName)s]: %(message)s")


class Processing2RNN:
    """
    转换RNN模型数据
    """

    def __init__(self):
        # 中间信息打印频率
        self.print_freq = 1
        # 序列长度
        self.seq_len = 100
        # 记录的所有动作数 主牌动作309个 + START,SEP,NONE(用于填充)
        self.num_record_mcard_action = 312
        # 主牌动作
        self.num_mcard_action = 309
        # 手牌索引
        self.card_index = "34567890JQKA2wW"
        # 所有手牌
        self.all_card = "33334444555566667777888899990000JJJJQQQQKKKKAAAA2222wW"

        # 所有主牌动作
        self.mcard_action = ['Solo_3', 'Solo_4', 'Solo_5', 'Solo_6', 'Solo_7', 'Solo_8', 'Solo_9', 'Solo_0', 'Solo_J',
                             'Solo_Q', 'Solo_K', 'Solo_A', 'Solo_2', 'Solo_w', 'Solo_W', 'SoloC_34567', 'SoloC_45678',
                             'SoloC_56789', 'SoloC_67890', 'SoloC_7890J', 'SoloC_890JQ', 'SoloC_90JQK', 'SoloC_0JQKA',
                             'SoloC_345678', 'SoloC_456789', 'SoloC_567890', 'SoloC_67890J', 'SoloC_7890JQ',
                             'SoloC_890JQK', 'SoloC_90JQKA', 'SoloC_3456789', 'SoloC_4567890', 'SoloC_567890J',
                             'SoloC_67890JQ', 'SoloC_7890JQK', 'SoloC_890JQKA', 'SoloC_34567890', 'SoloC_4567890J',
                             'SoloC_567890JQ', 'SoloC_67890JQK', 'SoloC_7890JQKA', 'SoloC_34567890J', 'SoloC_4567890JQ',
                             'SoloC_567890JQK', 'SoloC_67890JQKA', 'SoloC_34567890JQ', 'SoloC_4567890JQK',
                             'SoloC_567890JQKA', 'SoloC_34567890JQK', 'SoloC_4567890JQKA', 'SoloC_34567890JQKA',
                             'Pair_33', 'Pair_44', 'Pair_55', 'Pair_66', 'Pair_77', 'Pair_88', 'Pair_99', 'Pair_00', 'Pair_JJ',
                             'Pair_QQ', 'Pair_KK', 'Pair_AA', 'Pair_22', 'PairC_334455', 'PairC_445566', 'PairC_556677',
                             'PairC_667788', 'PairC_778899', 'PairC_889900', 'PairC_9900JJ', 'PairC_00JJQQ',
                             'PairC_JJQQKK', 'PairC_QQKKAA', 'PairC_33445566', 'PairC_44556677', 'PairC_55667788',
                             'PairC_66778899', 'PairC_77889900', 'PairC_889900JJ', 'PairC_9900JJQQ', 'PairC_00JJQQKK',
                             'PairC_JJQQKKAA', 'PairC_3344556677', 'PairC_4455667788', 'PairC_5566778899', 'PairC_6677889900',
                             'PairC_77889900JJ', 'PairC_889900JJQQ', 'PairC_9900JJQQKK', 'PairC_00JJQQKKAA',
                             'PairC_334455667788', 'PairC_445566778899', 'PairC_556677889900', 'PairC_6677889900JJ',
                             'PairC_77889900JJQQ', 'PairC_889900JJQQKK', 'PairC_9900JJQQKKAA', 'PairC_33445566778899',
                             'PairC_44556677889900', 'PairC_556677889900JJ', 'PairC_6677889900JJQQ',
                             'PairC_77889900JJQQKK', 'PairC_889900JJQQKKAA', 'PairC_3344556677889900', 'PairC_44556677889900JJ',
                             'PairC_556677889900JJQQ', 'PairC_6677889900JJQQKK', 'PairC_77889900JJQQKKAA',
                             'PairC_3344556677889900JJ', 'PairC_44556677889900JJQQ', 'PairC_556677889900JJQQKK',
                             'PairC_6677889900JJQQKKAA', 'PairC_3344556677889900JJQQ', 'PairC_44556677889900JJQQKK',
                             'PairC_556677889900JJQQKKAA', 'Trio_333', 'Trio_444', 'Trio_555', 'Trio_666', 'Trio_777',
                             'Trio_888', 'Trio_999', 'Trio_000', 'Trio_JJJ', 'Trio_QQQ', 'Trio_KKK', 'Trio_AAA', 'Trio_222',
                             'TrioC_333444', 'TrioC_444555', 'TrioC_555666', 'TrioC_666777', 'TrioC_777888',
                             'TrioC_888999', 'TrioC_999000', 'TrioC_000JJJ', 'TrioC_JJJQQQ', 'TrioC_QQQKKK', 'TrioC_KKKAAA',
                             'TrioC_333444555', 'TrioC_444555666', 'TrioC_555666777', 'TrioC_666777888',
                             'TrioC_777888999', 'TrioC_888999000', 'TrioC_999000JJJ', 'TrioC_000JJJQQQ', 'TrioC_JJJQQQKKK',
                             'TrioC_QQQKKKAAA', 'TrioC_333444555666', 'TrioC_444555666777', 'TrioC_555666777888',
                             'TrioC_666777888999', 'TrioC_777888999000', 'TrioC_888999000JJJ', 'TrioC_999000JJJQQQ',
                             'TrioC_000JJJQQQKKK', 'TrioC_JJJQQQKKKAAA', 'TrioC_333444555666777', 'TrioC_444555666777888',
                             'TrioC_555666777888999', 'TrioC_666777888999000', 'TrioC_777888999000JJJ', 'TrioC_888999000JJJQQQ',
                             'TrioC_999000JJJQQQKKK', 'TrioC_000JJJQQQKKKAAA', 'TrioC_333444555666777888',
                             'TrioC_444555666777888999', 'TrioC_555666777888999000', 'TrioC_666777888999000JJJ', 'TrioC_777888999000JJJQQQ',
                             'TrioC_888999000JJJQQQKKK', 'TrioC_999000JJJQQQKKKAAA', 'TrioK_333',
                             'TrioK_444', 'TrioK_555', 'TrioK_666', 'TrioK_777', 'TrioK_888', 'TrioK_999', 'TrioK_000', 'TrioK_JJJ',
                             'TrioK_QQQ', 'TrioK_KKK', 'TrioK_AAA', 'TrioK_222', 'TrioCK_333444',
                             'TrioCK_444555', 'TrioCK_555666', 'TrioCK_666777', 'TrioCK_777888', 'TrioCK_888999', 'TrioCK_999000',
                             'TrioCK_000JJJ', 'TrioCK_JJJQQQ', 'TrioCK_QQQKKK', 'TrioCK_KKKAAA',
                             'TrioCK_333444555', 'TrioCK_444555666', 'TrioCK_555666777', 'TrioCK_666777888', 'TrioCK_777888999',
                             'TrioCK_888999000', 'TrioCK_999000JJJ', 'TrioCK_000JJJQQQ', 'TrioCK_JJJQQQKKK',
                             'TrioCK_QQQKKKAAA', 'TrioCK_333444555666', 'TrioCK_444555666777', 'TrioCK_555666777888',
                             'TrioCK_666777888999', 'TrioCK_777888999000', 'TrioCK_888999000JJJ', 'TrioCK_999000JJJQQQ',
                             'TrioCK_000JJJQQQKKK', 'TrioCK_JJJQQQKKKAAA', 'TrioCK_333444555666777', 'TrioCK_444555666777888',
                             'TrioCK_555666777888999', 'TrioCK_666777888999000', 'TrioCK_777888999000JJJ',
                             'TrioCK_888999000JJJQQQ', 'TrioCK_999000JJJQQQKKK', 'TrioCK_000JJJQQQKKKAAA', 'TrioPK_333',
                             'TrioPK_444', 'TrioPK_555', 'TrioPK_666', 'TrioPK_777', 'TrioPK_888', 'TrioPK_999',
                             'TrioPK_000', 'TrioPK_JJJ', 'TrioPK_QQQ', 'TrioPK_KKK', 'TrioPK_AAA', 'TrioPK_222', 'TrioCPK_333444',
                             'TrioCPK_444555', 'TrioCPK_555666', 'TrioCPK_666777', 'TrioCPK_777888',
                             'TrioCPK_888999', 'TrioCPK_999000', 'TrioCPK_000JJJ', 'TrioCPK_JJJQQQ', 'TrioCPK_QQQKKK',
                             'TrioCPK_KKKAAA', 'TrioCPK_333444555', 'TrioCPK_444555666', 'TrioCPK_555666777',
                             'TrioCPK_666777888', 'TrioCPK_777888999', 'TrioCPK_888999000', 'TrioCPK_999000JJJ',
                             'TrioCPK_000JJJQQQ', 'TrioCPK_JJJQQQKKK', 'TrioCPK_QQQKKKAAA', 'TrioCPK_333444555666',
                             'TrioCPK_444555666777', 'TrioCPK_555666777888', 'TrioCPK_666777888999', 'TrioCPK_777888999000',
                             'TrioCPK_888999000JJJ', 'TrioCPK_999000JJJQQQ', 'TrioCPK_000JJJQQQKKK',
                             'TrioCPK_JJJQQQKKKAAA', 'BombK_3333', 'BombK_4444', 'BombK_5555', 'BombK_6666',
                             'BombK_7777', 'BombK_8888', 'BombK_9999', 'BombK_0000', 'BombK_JJJJ', 'BombK_QQQQ', 'BombK_KKKK',
                             'BombK_AAAA', 'BombK_2222', 'BombPK_3333', 'BombPK_4444', 'BombPK_5555', 'BombPK_6666',
                             'BombPK_7777', 'BombPK_8888', 'BombPK_9999', 'BombPK_0000', 'BombPK_JJJJ', 'BombPK_QQQQ',
                             'BombPK_KKKK', 'BombPK_AAAA', 'BombPK_2222', 'Bomb_3333', 'Bomb_4444', 'Bomb_5555',
                             'Bomb_6666', 'Bomb_7777', 'Bomb_8888', 'Bomb_9999', 'Bomb_0000', 'Bomb_JJJJ', 'Bomb_QQQQ',
                             'Bomb_KKKK', 'Bomb_AAAA', 'Bomb_2222', 'Rocket_wW', 'PASS']

        # 主牌动作历史记录动作列表
        self.record_mcard_action = ["None", "START", "SEP"]
        self.record_mcard_action.extend(self.mcard_action)

        # 主牌动作字符串转数字字典
        # 使用有限字典限定顺序
        self.mcard_action_to_num_dict = OrderedDict(((k, v) for v, k in enumerate(self.mcard_action)))
        # 主牌动作数字转字符串字典
        self.mcard_action_to_str_dict = OrderedDict(((k, v) for k, v in enumerate(self.mcard_action)))

        # 包含START SEP None 标签的主牌动作记录字典
        self.record_mcard_action_to_num_dict = OrderedDict(((k, v) for v, k in enumerate(self.record_mcard_action)))
        self.record_mcard_action_to_str_dict = OrderedDict(((k, v) for k, v in enumerate(self.record_mcard_action)))

        # 带牌类型字典
        record_ktype_action = ["None", "START", "SEP", "Solo", "Pair"]
        self.record_ktype_action_to_num_dict = OrderedDict(((k, v) for v, k in enumerate(record_ktype_action)))
        self.record_ktype_action_to_str_dict = OrderedDict(((k, v) for k, v in enumerate(record_ktype_action)))

        # 带牌长度字典
        record_knum_action = ["None", "START", "SEP", "1", "2", "3", "4", "5"]
        self.record_knum_action_to_num_dict = OrderedDict(((k, v) for v, k in enumerate(record_knum_action)))
        self.record_knum_action_to_str_dict = OrderedDict(((k, v) for k, v in enumerate(record_knum_action)))

    def padding_zero(self, record):
        """
        填充历史记录序列到指定长度
        :param record: 历史记录序列
        :return:
        """
        assert isinstance(record, list)
        zero_list = [0 for _ in range(self.seq_len - len(record))]
        record.extend(zero_list)
        return record

    def sort_card(self, cards):
        """
        对牌进行排序
        :param cards: 待排序的牌
        :return:返回排序后的牌
        """
        assert isinstance(cards, str)
        if cards == 'PASS':
            return cards
        card_temp = list(cards)
        card_temp.sort(key=lambda x: self.card_index.index(x))
        return ''.join(card_temp)

    def sub(self, cards1, cards2):
        """
        计算两组牌之间的差
        :param cards1: 第一组牌
        :param cards2: 第二组牌
        :return: 两组牌的差集
        """
        assert isinstance(cards1, str)
        assert isinstance(cards2, str)

        d = Differ()

        # 断言一组牌必定包含在另外一组牌当中
        assert len(set([string[0] for string in d.compare(cards1, cards2) if string[0] in ["-", "+"]])) <= 1

        if len(cards1) >= len(cards2):
            remain = [j[-1] for j in filter(lambda s: s[0] == '-', d.compare(cards1, cards2))]
        else:
            remain = [j[-1] for j in filter(lambda s: s[0] == '+', d.compare(cards1, cards2))]
        return self.sort_card("".join(remain))

    def check_total_cards(self, output_cards, hands, down_hands, up_hands):
        """
        检查总牌数是否相等
        :param output_cards:已经打算的牌
        :param hands:当前手牌
        :param down_hands:下家手牌
        :param up_hands:上家手牌
        :return:bool
        """
        assert isinstance(output_cards, str)
        assert isinstance(hands, str)
        assert isinstance(down_hands, str)
        assert isinstance(up_hands, str)

        # 如果之前没有出过牌
        if output_cards != "None":
            total_cards = self.sort_card(output_cards + hands + down_hands + up_hands)
        else:
            total_cards = self.sort_card(hands + down_hands + up_hands)

        return total_cards == self.all_card

    def check_up_down_hands(self, hidden_cards, down_hands, up_hands):
        """
        检查上家牌和下家牌是否等于当前未知牌
        :param hidden_cards: 未知牌
        :param down_hands: 下家牌
        :param up_hands: 上家牌
        :return: bool
        """
        assert isinstance(hidden_cards, str)
        assert isinstance(down_hands, str)
        assert isinstance(up_hands, str)

        up_down_total_hands = self.sort_card(down_hands + up_hands)

        return hidden_cards == up_down_total_hands

    def ktype_action_transform(self, action):
        """
        如果带牌类型是字符串则转换为数字,如果带牌类型是数字则转换为字符串
        :param action: 动作
        :return:
        """
        if isinstance(action, str):
            return self.record_ktype_action_to_num_dict[action]
        else:
            return self.record_ktype_action_to_str_dict[action]

    def klen_action_transform(self, action):
        """
        如果带牌长度是字符串则转换为数字
        如果带牌长度是数字则转换为字符串
        :param action: 动作
        :return:
        """
        if isinstance(action, str):
            return self.record_knum_action_to_num_dict[action]
        else:
            return self.record_knum_action_to_str_dict[action]

    def mcard_action_transform(self, action):
        """
        如果主牌动作是字符串,则返回数字
        如果主牌动作是数据则返回字符串
        :param action:
        :return:
        """
        if isinstance(action, str):
            return self.mcard_action_to_num_dict[action]
        else:
            return self.mcard_action_to_str_dict[action]

    def record_mcard_action_transform(self, action):
        """
        这个里面有312种动作
        如果主牌动作是字符串,则返回数字
        如果主牌动作是数据则返回字符串
        :param action:
        :return:
        """
        if isinstance(action, str):
            return self.record_mcard_action_to_num_dict[action]
        else:
            return self.record_mcard_action_to_str_dict[action]

    def read_csv(self, file_path):
        """
        读取CSV文件
        :param file_path: 文件路径
        :return: pd.DataFrame
        """
        assert isinstance(file_path, str)

        logging.debug("正在读取CSV数据")
        read_data = pd.read_csv(filepath_or_buffer=file_path, encoding="utf-8", header=0)
        read_data["game_session"] = np.cumsum(read_data.record == "None")

        logging.debug("读取CSV数据完成")

        return read_data

    def get_one_sessoin_data(self, data, session_id):
        """
        获取一场比赛的游戏数据
        :param data: 全部游戏数据
        :param session_id:场次索引
        :return: one_session_pid_data 玩家角色数据
                one_session_mcard_data 玩家主牌数据
                one_session_kcard_data 玩家带牌数据
                one_session_hidden_data 未知牌数据
                one_session_hands_data 玩家手牌数据
                one_session_output_data 已经打出的牌数据
                one_session_record_data 之前打牌的历史记录
        """

        assert isinstance(data, pd.DataFrame)

        if session_id % self.print_freq == 0:
            string = "正在获取第{session_id}场数据".format(session_id=session_id)
            logging.debug(string)

        # 单场游戏的数据
        one_session_data = data[data.game_session == session_id]

        # 玩家编号
        # 0 表示庄家
        # 1 表示下家
        # 2 表示上家
        one_session_pid_data = list(one_session_data.pid)

        # 因为有些地方pid不规范，需要做一下调整
        if one_session_pid_data[0] == 1:
            one_session_pid_data = [num - 1 for num in one_session_pid_data]

        # 获取主牌数据
        one_session_mcard_data = list(one_session_data.mcard)
        # 获取带牌数据
        one_session_kcard_data = list(one_session_data.kcard)
        # 获取手牌数据
        one_session_hands_data = list(one_session_data.hands)
        # 获取未知牌数据
        one_session_hidden_data = list(one_session_data.hidden)
        # 获取已经打出牌的数据
        one_session_output_data = list(one_session_data.output)

        if session_id % self.print_freq == 0:
            string = "第{session_id}场数据读取完成".format(session_id=session_id)
            logging.debug(string)

        result = {
            "one_session_pid_data": one_session_pid_data,
            "one_session_mcard_data": one_session_mcard_data,
            "one_session_kcard_data": one_session_kcard_data,
            "one_session_hidden_data": one_session_hidden_data,
            "one_session_hands_data": one_session_hands_data,
            "one_session_output_data": one_session_output_data
        }

        return result

    def kcard_message_transform(self, kcard_message):
        """
        对于带牌信息的处理
        :param kcard_message: string 表示带牌的信息
        :return: 返回带牌的信息
        """
        assert isinstance(kcard_message, str)
        if kcard_message == "None":  # 没有从牌的情况处理
            return ["None", "None"]
        else:
            # 有从牌的情况
            # 因为不同的牌(单牌或对牌)之间使用";"间隔
            # 可以使用这个提取信息
            kcard_message_split = kcard_message.split(sep=";")

            # 不同牌的种类数量
            kcard_num = len(kcard_message_split)

            min_num = min([len(list(_)) for _ in kcard_message_split])
            max_num = max([len(list(_)) for _ in kcard_message_split])

            # 如果不成立，则说明数据存在问题
            assert min_num == max_num

            if min_num == 1:
                return ["Solo", kcard_num]
            else:
                return ["Pair", kcard_num]

    def one_session_data_processing_to_csv(self, session_id, data):
        """
        将一场比赛数据进行处理以适应CSV形式保存
        :param session_id:游戏场次号
        :param data:
        :return:
        """
        """
        读取一场游戏数据
        """

        one_session_data = self.get_one_sessoin_data(data=data, session_id=session_id)
        one_session_pid_data = one_session_data["one_session_pid_data"]
        one_session_mcard_data = one_session_data["one_session_mcard_data"]
        one_session_kcard_data = one_session_data["one_session_kcard_data"]
        one_session_hidden_data = one_session_data["one_session_hidden_data"]
        one_session_hands_data = one_session_data["one_session_hands_data"]
        one_session_output_data = one_session_data["one_session_output_data"]

        # 打印一下当前正在处理的场次
        if session_id % self.print_freq == 0:
            string = "正在处理第{session_id}场数据".format(session_id=session_id)
            logging.debug(string)

        # 历史记录
        # START 起始标志
        mcard_record = ["START"]  # 主牌的历史记录
        # 第一个位置表示牌型
        # 第二个位置表示牌数量
        ktype_record = ["START"]  # 带牌的历史记录
        klen_record = ["START"]

        one_session_game_record = []

        for game_step in range(len(one_session_pid_data)):

            """
           记录前两个回合的打牌数据,用于推断当前打牌类型
           """
            if game_step >= 1:
                tmp_mcard_two_time_before_record.append(one_session_mcard_data[game_step - 1])
                del tmp_mcard_two_time_before_record[0]

            else:
                tmp_mcard_two_time_before_record = ["None", "None"]
            mcard_two_time_before_record = ";".join(tmp_mcard_two_time_before_record)

            """
            记录上家和下家的手牌信息,用于做数据预测
            """
            if game_step < len(one_session_pid_data) - 2:
                down_hands = one_session_hands_data[game_step + 1]
                up_hands = one_session_hands_data[game_step + 2]
            # 倒数第二场
            elif game_step == len(one_session_pid_data) - 2:
                down_hands = one_session_hands_data[game_step + 1]
                up_hands = self.sub(cards1=one_session_hidden_data[game_step], cards2=down_hands)
                tmp_up_hands = up_hands
            # 最后一场
            else:
                down_hands = tmp_up_hands
                up_hands = self.sub(cards1=one_session_hidden_data[game_step], cards2=down_hands)

            """
            处理带牌信息
            """
            kcard_message = one_session_kcard_data[game_step]
            kcard_type, kcard_num = self.kcard_message_transform(kcard_message)

            """
            将处理完的数据打包,并保存成one_session_game_record中
            """
            one_session_game_record.append([session_id,  # 场次id
                                            one_session_pid_data[game_step],  # 玩家角色id
                                            mcard_record.copy(),  # 主牌历史记录
                                            ktype_record.copy(),  # 带牌类型历史记录
                                            klen_record.copy(),  # 带牌张数历史记录
                                            mcard_two_time_before_record,  # 之前两次的出牌记录
                                            one_session_output_data[game_step],  # 已经打出牌的记录
                                            one_session_hidden_data[game_step],  # 未知牌的记录
                                            one_session_hands_data[game_step],  # 玩家手牌记录
                                            down_hands,  # 下家手牌记录
                                            up_hands,  # 上家手牌记录
                                            len(one_session_output_data[game_step]) if one_session_output_data[game_step] != "None" else 0,  # 已经出牌的数量
                                            len(one_session_hidden_data[game_step]),  # 未知牌的数量
                                            len(one_session_hands_data[game_step]),  # 玩家手牌数量记录
                                            len(down_hands),  # 下家手牌数量记录
                                            len(up_hands),  # 上家手牌数量记录
                                            one_session_mcard_data[game_step],  # 玩家打牌的主牌记录
                                            kcard_type,  # 玩家出牌的带牌类型记录
                                            kcard_num])  # 玩家带牌数量记录

            """
            更新下一回合数据
            """
            mcard_record.append(one_session_mcard_data[game_step])

            ktype_record.append(kcard_type)
            klen_record.append(str(kcard_num))

            """
            插入SEP 每一轮插入一个"SEP" 表示经历过一轮
            """
            if (game_step + 1) % 3 == 0:
                mcard_record.append("SEP")
                ktype_record.append("SEP")
                klen_record.append("SEP")

        if session_id % self.print_freq == 0:
            string = "第{session_id}场数据处理完成".format(session_id=session_id)
            logging.debug(string)

        return one_session_game_record

    def get_cards_all_action(self, cards):
        """
        当前牌组的所有可出动作
        :param cards: 牌组
        :return:
        """

        if cards == "None":
            return ["PASS"]
        else:
            cards_all_action = set()
            counter = Counter(cards)
            for i in range(3, len(self.mcard_action_to_num_dict)):
                mcard_action = self.mcard_action_transform(action=i)
                if mcard_action == "PASS":
                    cards_all_action.add("PASS")
                else:
                    mcard_type, mcards = mcard_action.split("_")
                    if self.mcard_in_hands(mcards=mcards, hands=cards):
                        if mcard_type == "TrioK":
                            # 对于三带一,要求总牌数必须超过四张,且有两种以上牌型
                            if len(cards) >= 4 and len(counter) >= 2:
                                cards_all_action.add(mcard_action)
                        elif mcard_type == "TrioPK":
                            # 对于三带一对, 要求总牌数必须5张以上,且有两种牌数量大于2
                            if len(cards) >= 5 and self.get_above_n_num(cards, 2) >= 2:
                                cards_all_action.add(mcard_action)
                        elif mcard_type == "TrioCK":
                            # 对于带单的飞机,每三个加一张单
                            min_length = len(Counter(mcards))
                            if len(cards) >= min_length * 4:
                                cards_all_action.add(mcard_action)
                        elif mcard_type == "TrioCPK":
                            # 对于飞机带对牌
                            min_length = len(Counter(mcards))
                            if len(cards) > min_length * 5 and self.get_above_n_num(cards=cards, n=2) >= min_length * 2:
                                cards_all_action.add(mcard_action)
                        elif mcard_type == "BombK":
                            if len(cards) >= 6:
                                cards_all_action.add(mcard_action)
                        elif mcard_type == "BombPK":
                            if len(cards) >= 8 and (self.get_above_n_num(cards=cards, n=2) >= 3 or self.get_above_n_num(cards=cards, n=4) >= 2):
                                cards_all_action.add(mcard_action)
                        else:
                            cards_all_action.add(mcard_action)
        return cards_all_action

    def mcard_action2vec(self, mcard):
        """
        将主牌动作转换为309维向量
        :param mcard:主牌动作
        :return:
        """
        mcard_index = self.mcard_action_transform(action=mcard)
        mcard_action_vec = [0 for _ in range(self.num_mcard_action)]
        mcard_action_vec[mcard_index] = 1
        return np.array(mcard_action_vec)

    def cards_all_action2vec(self, cards_all_action):
        """
        根据当前牌求出所有可执行动作,并将这个动作用向量表示,共309维
        :param cards_all_action:
        :return:
        """
        cards_all_action_vec = [0 for _ in range(self.num_mcard_action)]
        for mcard_action in cards_all_action:
            cards_all_action_vec += self.mcard_action2vec(mcard=mcard_action)
        return cards_all_action_vec

    def mcard_in_hands(self, mcards, hands):
        """
        判断当前主牌动作在当前手牌下是不是可出
        :param mcards:主牌动作
        :param hands:手牌
        :return:
        """
        d = Differ()
        mcards = self.sort_card(cards=mcards)
        hands = self.sort_card(cards=hands)

        for s in d.compare(a=hands, b=mcards):
            if s[0] == "+":
                return False
        return True

    def get_above_n_cards(self, cards, n):
        """
        统计手牌中张数多于n的牌
        :param cards: 手牌
        :param n: 数量
        :return:
        """
        counter = Counter(cards)
        return [card for card, num in counter.items() if num >= n]

    def get_above_n_num(self, cards, n):
        """
        统计手牌中数量多于n的手牌的数量
        :param cards: 手牌
        :param n: 数量
        :return:
        """
        return len(self.get_above_n_cards(cards=cards, n=n))

    def card2vec(self, cards):
        """
        将卡牌数据转换成向量
        0个A表示成[1,0,0,0,0]
        1个A用于成[0,1,0,0,0]
        将不同类型手牌拼接,最终为69维向量
        :param cards:手牌
        :return:
        """
        hands_counter = Counter(cards)
        result = []
        for card_type in self.card_index:
            if card_type not in "wW":
                tmp = [0, 0, 0, 0, 0]
                if card_type in hands_counter.keys():
                    tmp[hands_counter[card_type]] = 1
                else:
                    tmp[0] = 1
            else:
                tmp = [0, 0]
                if card_type in hands_counter.keys():
                    tmp[hands_counter[card_type]] = 1
                else:
                    tmp[0] = 1
            result.extend(tmp)
        return np.array(result)

    def one_session_data_processing_to_pickle(self, session_id, data):
        """
        对一场比赛数据进行处理,处理之后数据用于保存成pickle文件
        :param session_id:场次编号
        :param data:数据
        :return:
        """
        one_session_data = self.get_one_sessoin_data(data=data, session_id=session_id)
        one_session_pid_data = one_session_data["one_session_pid_data"]
        one_session_mcard_data = one_session_data["one_session_mcard_data"]
        one_session_kcard_data = one_session_data["one_session_kcard_data"]
        one_session_hidden_data = one_session_data["one_session_hidden_data"]
        one_session_hands_data = one_session_data["one_session_hands_data"]
        one_session_output_data = one_session_data["one_session_output_data"]

        if session_id % self.print_freq == 0:
            string = "正在处理第{session_id}场数据".format(session_id=session_id)
            logging.debug(string)

        # 历史记录
        # START 起始标志
        mcard_record = [self.record_mcard_action_transform(action="START")]  # 主牌的历史记录
        # 第一个位置表示牌型
        # 第二个位置表示牌数量
        ktype_record = [self.ktype_action_transform(action="START")]  # 带牌的历史记录
        klen_record = [self.klen_action_transform(action="START")]

        one_session_game_record = []

        for game_step in range(len(one_session_pid_data)):

            # 之前两场的游戏记录
            if game_step >= 1:
                mcard_two_time_before_record.append(self.record_mcard_action_transform(one_session_mcard_data[game_step - 1]))
                del mcard_two_time_before_record[0]

            else:
                mcard_two_time_before_record = [self.record_mcard_action_transform("None"), self.record_mcard_action_transform("None")]

            """
            计算每个玩家手牌信息
            """
            # 对于从第一回合到到数第二回合的数据处理
            if game_step < len(one_session_pid_data) - 2:
                one_session_down_hands = one_session_hands_data[game_step + 1]
                one_session_up_hands = one_session_hands_data[game_step + 2]
            # 倒数第二场
            elif game_step == len(one_session_pid_data) - 2:
                one_session_down_hands = one_session_hands_data[game_step + 1]
                one_session_up_hands = self.sub(cards1=one_session_hidden_data[game_step], cards2=one_session_down_hands)
                tmp_one_session_up_hands = one_session_up_hands
            # 最后一场
            else:
                one_session_down_hands = tmp_one_session_up_hands
                one_session_up_hands = self.sub(cards1=one_session_hidden_data[game_step], cards2=one_session_down_hands)

            # 对带牌信息进行处理
            kcard_message = one_session_kcard_data[game_step]
            kcard_type, kcard_num = self.kcard_message_transform(kcard_message)

            # 将求出未知牌所有动作并转换为向量
            hidden_all_action_vec = self.cards_all_action2vec(self.get_cards_all_action(one_session_hidden_data[game_step]))
            # 求出已当前玩家手牌可执行动作并转换为向量
            hands_all_action_vec = self.cards_all_action2vec(self.get_cards_all_action(one_session_hands_data[game_step]))

            game_record_dict = {
                "session_id": session_id,
                "gamer_id": one_session_pid_data[game_step],
                "mcard_action_record": self.padding_zero(mcard_record.copy()),
                "ktype_action_record": self.padding_zero(ktype_record.copy()),
                "klen_action_record": self.padding_zero(klen_record.copy()),
                "mcard_record_2s_before": mcard_two_time_before_record.copy(),
                "output_cards_vec": self.card2vec(one_session_output_data[game_step]),
                "hidden_cards_vec": self.card2vec(one_session_hidden_data[game_step]),
                "hands_vec": self.card2vec(one_session_hands_data[game_step]),
                "down_hands_vec": self.card2vec(one_session_down_hands),
                "up_hands_vec": self.card2vec(one_session_up_hands),
                "hidden_cards_action_vec": hidden_all_action_vec,
                "inf_hidden_cards_action_vec": [-np.inf if _ == 0 else 0 for _ in hidden_all_action_vec],
                "hands_action_vec": hands_all_action_vec,
                "inf_hands_action_vec": [-np.inf if _ == 0 else 0 for _ in hands_all_action_vec],
                "num_output_cards": len(one_session_output_data[game_step]) if one_session_output_data[game_step] != "None" else 0,
                "num_hidden_cards": len(one_session_hidden_data[game_step]),
                "num_hands": len(one_session_hands_data[game_step]),
                "num_up_hands": len(one_session_up_hands),
                "num_down_hands": len(one_session_down_hands),
                "next_mcard_action_label": self.mcard_action_transform(one_session_mcard_data[game_step])
            }

            one_session_game_record.append(game_record_dict)

            # 更新下一回合游戏数据
            mcard_record.append(self.record_mcard_action_transform(one_session_mcard_data[game_step]))

            ktype_record.append(self.ktype_action_transform(kcard_type))
            klen_record.append(self.klen_action_transform(str(kcard_num)))

            # 每一轮插入一个"SEP"
            # 表示经历过一轮
            if (game_step + 1) % 3 == 0:
                mcard_record.append(self.record_mcard_action_transform("SEP"))
                ktype_record.append(self.ktype_action_transform("SEP"))
                klen_record.append(self.klen_action_transform("SEP"))

        if session_id % self.print_freq == 0:
            string = "第{session_id}场数据处理完成".format(session_id=session_id)
            logging.debug(string)

        return one_session_game_record

    def processing_data_to_csv(self, data):
        """
        对数据进行预处理，返回处理结果
        :param data: 原始数据
        :return: 处理后数据
        """
        num_session = max(data.game_session)

        result = []

        for session_id in range(1, num_session + 1):
            one_session_game_record = self.one_session_data_processing_to_csv(session_id=session_id, data=data)
            result.extend(one_session_game_record)

        # result = []
        #
        # num_session = len(tmp_game_record)
        #
        # for session_id in range(num_session):
        #     result.extend(tmp_game_record[session_id])

        result = [[session_id,  # 游戏场次记录
                   gamer_id,  # 玩家角色id记录
                   ";".join(mcard_record),  # 历史主牌记录
                   ";".join(ktype_record),  # 历史带牌类型记录
                   ";".join(klen_record),  # 历史带牌数量记录
                   mcard_two_time_before_record,
                   output_cards,  # 已经打出的牌的记录
                   hidden_cards,  # 未知牌记录
                   hands,  # 玩家手牌记录
                   down_hands,  # 玩家手牌记录
                   up_hands,  # 上家手牌记录
                   num_output_cards,
                   num_hidden_cards,
                   num_hands,
                   num_down_hands,
                   num_up_hands,
                   mcard_play,  # 玩家打出的主牌
                   ktype_play,  # 玩家的带牌类型
                   klen_play,
                   self.check_total_cards(output_cards=output_cards,
                                          hands=hands,
                                          down_hands=down_hands,
                                          up_hands=up_hands) and
                   self.check_up_down_hands(hidden_cards=hidden_cards,
                                            down_hands=down_hands,
                                            up_hands=up_hands)] for  # 玩家的带牌数量
                  session_id,
                  gamer_id,
                  mcard_record,
                  ktype_record,
                  klen_record,
                  mcard_two_time_before_record,
                  output_cards,
                  hidden_cards,
                  hands,
                  down_hands,
                  up_hands,
                  num_output_cards,
                  num_hidden_cards,
                  num_hands,
                  num_down_hands,
                  num_up_hands,
                  mcard_play,
                  ktype_play,
                  klen_play in result]

        result = pd.DataFrame(result)

        result.columns = ["session_id",
                          "gamer_id",
                          "mcard_record",
                          "ktype_record",
                          "klen_record",
                          "mcard_two_time_before_record",
                          "output_cards",
                          "hidden_cards",
                          "hands",
                          "down_hands",
                          "up_hands",
                          "num_output_cards",
                          "num_hidden_cards",
                          "num_hands",
                          "num_down_hands",
                          "num_up_hands",
                          "mcard_play",
                          "ktype_play",
                          "klen_play",
                          "is_qualified"]

        return result

    # def processing_data_to_pickle(self, data):
    #     """
    #     将所有数据进行预处理并保存成pickle文件
    #     :param data:
    #     :return:
    #     """
    #     num_session = max(data.game_session)
    #
    #     print(num_session)
    #
    #     for session_id in range(1, num_session + 1):
    #         one_session_game_record = self.one_session_data_processing_to_pickle(session_id=session_id, data=data)
    #         result.extend(one_session_game_record)
    #
    #     return result
    #
    # def save_to_pickle(self, data, save_path, num_record_of_one_pickle):
    #     # 统计记录的条数
    #     num_record = len(data)
    #     # 数据分割索引
    #     data_seq = [i for i in np.arange(start=0, stop=num_record, step=num_record_of_one_pickle)]
    #
    #     # 设置保存文件路径
    #
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #
    #     [os.remove(os.path.join(save_path, file_name)) for file_name in os.listdir(save_path)]
    #
    #     # 按之前索引分割文件
    #     for i in range(len(data_seq)):
    #         if i < len(data_seq) - 1:
    #             logging.debug("正在保存第{start}到{end}条数据".format(start=data_seq[i], end=data_seq[i + 1]))
    #             result_one_slice = data[data_seq[i]:data_seq[i + 1]]
    #         else:
    #             logging.debug("正在保存第{start}到{end}条数据".format(start=data_seq[i], end=len(data)))
    #             result_one_slice = data[data_seq[i]:]
    #
    #         save_file_name = "".join([save_path, "/", save_path, "_", str(i), ".pickle"])
    #
    #         # 保存数据
    #         with open(file=save_file_name, mode="wb") as f:
    #             pickle.dump(result_one_slice, f)
    #         logging.debug("保存全部pickle数据完成")

    def processing_and_save_to_pickle(self, save_file_name, data, session_batch=5000):
        num_session = max(data.game_session)
        file_index = 0
        num_index = 0
        result = []
        for session_id in range(30001, num_session + 1):
            one_session_game_record = self.one_session_data_processing_to_pickle(session_id=session_id, data=data)
            result.extend(one_session_game_record)
            num_index += 1

            if num_index == session_batch:
                num_index = 0
                with open(file=save_file_name + "_" + str(file_index) + ".pickle", mode="wb") as f:
                    pickle.dump(result, f)
                result = []
                file_index += 1
                logging.debug("第" + str(file_index) + "份数据保存完毕")

        with open(file=save_file_name + "_" + str(file_index) + ".pickle", mode="wb") as f:
            pickle.dump(result, f)

    def save_to_csv(self, data, save_path):
        """
        将数据保存成CSV格式
        :param data: 数据
        :param save_path: 保存路径
        :return: None
        """
        assert isinstance(data, pd.DataFrame)

        logging.debug("正在保存CSV数据")

        data.to_csv(path_or_buf=save_path, index=None)

        logging.debug("保存CSV数据完成")


if __name__ == "__main__":
    read_file_path = "sample10w.csv"
    csv_save_path = "sample10w_processing.csv"
    process2rnn = Processing2RNN()
    sample10w = process2rnn.read_csv(file_path=read_file_path)
    # process2rnn.one_session_data_processing_to_pickle(data=sample, session_id=1)

    # processing_data_pickle = process2rnn.processing_data_to_pickle(data=sample10w)
    #
    # process2rnn.save_to_pickle(data=processing_data_pickle, save_path="sample10w", num_record_of_one_pickle=5000)

    process2rnn.processing_and_save_to_pickle(save_file_name="sample100w/sample100w", data=sample10w, session_batch=5000)
