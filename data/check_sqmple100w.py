# -*- coding: utf-8 -*-

from data.loader import Loader

path = "sample100w"

loader = Loader(path=path)

flag = True
for one_data in loader.read_pickle():
    flag = flag and one_data["hands_action_vec"][one_data["next_mcard_action_label"]] == 1
    if flag is False:
        break

print(flag)
