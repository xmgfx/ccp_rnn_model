# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np


class Loader(object):

    def __init__(self, path):
        self.path = path

    def _create_record_dict(self):
        keys = ['session_id', 'gamer_id', 'mcard_action_record', 'ktype_action_record', 'klen_action_record',
                'mcard_record_2s_before', 'output_cards_vec', 'hidden_cards_vec', 'hands_vec', 'down_hands_vec',
                'up_hands_vec', 'hidden_cards_action_vec', 'inf_hidden_cards_action_vec', 'hands_action_vec',
                'inf_hands_action_vec', 'num_output_cards', 'num_hidden_cards', 'num_hands', 'num_up_hands',
                'num_down_hands', 'next_mcard_action_label']
        values = [[] for _ in range(len(keys))]

        record_dict = {k: v for k, v in zip(keys, values)}

        return record_dict

    def read_pickle(self):
        for file_name in os.listdir(self.path):
            file_path = os.path.join(self.path, file_name)
            one_pickle_file_read = pd.read_pickle(file_path)
            for one_record in one_pickle_file_read:
                yield one_record

    def read_batch(self, batch_size):
        record_dict = self._create_record_dict()
        i = 1
        for one_record in self.read_pickle():
            for k in record_dict.keys():
                record_dict[k].append(one_record[k])
            if i == batch_size:
                yield record_dict
                i = 1
                record_dict = self._create_record_dict()
            i += 1
        yield record_dict


if __name__ == "__main__":
    path = "sample100w"
    loader = Loader(path=path)
    for tmp in loader.read_batch(batch_size=10):
        print(tmp)
        break
