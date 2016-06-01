#!/usr/bin/env python3

import pandas

def gen_ans(res, ID, classes, file_name):
    res_num = len(res)
    ID_num = len(ID)
    if (res_num != ID_num):
        res = pandas.Series(res).reshape(ID_num, res_num // ID_num)

    res_data = pandas.DataFrame()
    res_data['ID'] = ID
    for class_, i in zip(classes, range(0, len(classes))):
        res_data[class_] = res[:,i]

    res_data.to_csv(file_name, index=False)
    return 0
