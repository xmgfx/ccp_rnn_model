# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from collections import namedtuple
from collections import deque
from common import deck
from common import play_card_action
from common import play_card_record

# 配置调试
logging.basicConfig(level=logging.DEBUG, format="[%(threadName)s]: %(message)s")


def read_csv(file_path):
    assert isinstance(file_path, str)
    logging.debug("正在读取CSV数据")
    read_data = pd.read_csv(filepath_or_buffer=file_path, encoding="utf-8", header=0)
    read_data["game_session"] = np.cumsum(read_data.record == "None")

    logging.debug("读取CSV数据完成")
    return read_data


def get_one_session_data(data, session_id, print_freq=200):
    assert isinstance(data, pd.DataFrame)
    if session_id % print_freq == 0:
        string = "正在获取第{session_id}场数据".format(session_id=session_id)
        logging.debug(string)
    # 单场游戏的数据
    one_session_data = data[data.game_session == session_id]

    result = OneSessionData()
    result.session_id = session_id
    result.pid = list(one_session_data.pid)
    result.mcard_action = list(one_session_data.mcard)
    result.kcard_action = list(one_session_data.kcard)
    result.current_player_deck = list(one_session_data.hands)
    result.hidden_deck = list(one_session_data.hidden)
    result.output_deck = list(one_session_data.output)

    if session_id % print_freq == 0:
        string = "第{session_id}场数据读取完成".format(session_id=session_id)
        logging.debug(string)

    return result


class SelfBuildClass(object):
    mcard_action = None
    record_mcard_action = None
    record_ktype_action = None
    record_klen_action = None

    def __init__(self):
        self.mcard_action = play_card_action.MCardAction()
        record_mcard_action = ["None", "START", "SEP"]
        record_mcard_action.extend(self.mcard_action.all_action)
        self.record_mcard_action = play_card_record.PlayCardRecord(all_action=record_mcard_action)
        record_ktype_action = ["None", "START", "SEP", "Solo", "Pair"]
        self.record_ktype_action = play_card_record.PlayCardRecord(all_action=record_ktype_action)
        record_klen_action = ["None", "START", "SEP", "1", "2", "3", "4", "5"]
        self.record_klen_action = play_card_record.PlayCardRecord(all_action=record_klen_action)

        self.deck = deck.Deck()


class OneSessionData(object):
    session_id = None
    pid = None
    mcard_action = None
    kcard_action = None
    current_player_deck = None
    hidden_deck = None
    output_deck = None

    def __init__(self):
        self.sbc = SelfBuildClass()

    def kcard_message_processing(self, kcard_message):

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

    def processing(self):
        # 历史记录
        # START 起始标志
        mcard_action_record = ["START"]  # 主牌的历史记录
        # 第一个位置表示牌型
        # 第二个位置表示牌数量
        ktype_action_record = ["START"]  # 带牌的历史记录
        klen_action_record = ["START"]

        result = []

        PlayCardsRecord = namedtuple(typename="PlayCardsRecord", field_names=["session_id",
                                                                              "gamer_id",
                                                                              "mcard_action_record",
                                                                              "ktype_action_record",
                                                                              "klen_action_record",
                                                                              "mcard_action_2s_before_record",
                                                                              "current_player_deck_vec",
                                                                              "output_deck_vec",
                                                                              "hidden_deck_vec",
                                                                              "current_player_deck_all_action_vec",
                                                                              "hidden_deck_all_action_vec",
                                                                              "down_player_deck",
                                                                              "up_player_deck"])

        mcard_2s_before_record = deque(maxlen=2)
        for game_step in range(len(self.pid)):
            """
           记录前两个回合的打牌数据,用于推断当前打牌类型
           """
            if game_step >= 1:
                mcard_action = self.mcard_action[game_step - 1]
                mcard_2s_before_record.append(self.sbc.mcard_action.action_to_num(action=mcard_action))
                # del tmp_mcard_2s_before_record[0]

            else:
                mcard_action = "PASS"
                mcard_2s_before_record.extend([self.sbc.mcard_action.action_to_num(action=mcard_action),
                                               self.sbc.mcard_action.action_to_num(action=mcard_action)])

            # mcard_2s_before_record = ";".join(mcard_2s_before_record)

            """
            记录上家和下家的手牌信息,用于做数据预测
            """
            if game_step < len(self.pid) - 2:
                down_palyer_deck = self.current_player_deck[game_step + 1]
                up_player_deck = self.current_player_deck[game_step + 2]
            # 倒数第二场
            elif game_step == len(self.pid) - 2:
                down_palyer_deck = self.current_player_deck[game_step + 1]
                up_player_deck = self.sbc.deck.sub_deck(deck1=self.hidden_deck[game_step], deck2=down_palyer_deck)
                tmp_up_player_deck = up_player_deck
            # 最后一场
            else:
                down_player_deck = tmp_up_player_deck
                up_player_deck = self.sbc.deck.sub_deck(deck1=self.hidden_deck[game_step], deck2=down_player_deck)
            """
            处理带牌信息
            """
            kcard_message = self.kcard_action[game_step]
            kcard_type, kcard_num = self.kcard_message_processing(kcard_message=kcard_message)

            mcard_action_record_seq = self.sbc.record_mcard_action.record2seq(record=mcard_action_record,
                                                                              seq_len=54,
                                                                              padding_zero=False,
                                                                              padding_zero_position="backward")

            ktype_action_record_seq = self.sbc.record_ktype_action.record2seq(record=ktype_action_record,
                                                                              seq_len=54,
                                                                              padding_zero=False,
                                                                              padding_zero_position="backward")

            klen_action_record_seq = self.sbc.record_klen_action.record2seq(record=klen_action_record,
                                                                            seq_len=54,
                                                                            padding_zero=False,
                                                                            padding_zero_position="backward")

            current_player_deck_vec = self.sbc.deck.deck2vec(deck=self.current_player_deck[game_step])
            output_deck_vec = self.sbc.deck.deck2vec(deck=self.output_deck[game_step])
            hidden_deck_vec = self.sbc.deck.deck2vec(deck=self.hidden_deck[game_step])
            current_player_deck_all_action_vec = self.sbc.deck.get_all_mcard_action_in_deck(deck=self.current_player_deck[game_step],
                                                                                            return_vec=False)

            hidden_deck_all_action_vec = self.sbc.deck.get_all_mcard_action_in_deck(deck=self.hidden_deck[game_step],
                                                                                    return_vec=False)

            play_card_record_info = PlayCardsRecord(self.session_id,
                                                    self.pid[game_step],
                                                    mcard_action_record_seq,
                                                    ktype_action_record_seq,
                                                    klen_action_record_seq,
                                                    list(mcard_2s_before_record),
                                                    current_player_deck_vec,
                                                    output_deck_vec,
                                                    hidden_deck_vec,
                                                    current_player_deck_all_action_vec,
                                                    hidden_deck_all_action_vec,
                                                    down_palyer_deck,
                                                    up_player_deck)
            result.append(play_card_record_info)

            mcard_action_record.append(self.mcard_action[game_step])
            ktype_action_record.append(kcard_type)
            klen_action_record.append(str(kcard_num))

            """
            插入SEP 每一轮插入一个"SEP" 表示经历过一轮
            """
            if (game_step + 1) % 3 == 0:
                mcard_action_record.append("SEP")
                ktype_action_record.append("SEP")
                klen_action_record.append("SEP")

        return result


if __name__ == "__main__":
    file_path = "sample.csv"
    read_data = read_csv(file_path=file_path)
    one_session_data = get_one_session_data(data=read_data, session_id=1)
    print(one_session_data.processing())
