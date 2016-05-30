#!/usr/bin/env python3

import pandas
import random
import os

def mk_nfold(train_file, cv=5):
    os.system('mkdir folds')
    train_handler = open(train_file, 'r')
    schema = train_handler.readline()
    data = train_handler.readlines()
    train_handler.close()

    for i in range(cv):
        train_part = open('folds/{}_train.csv'.format(str(i)), 'w')
        test_part = open('folds/{}_test.csv'.format(str(i)), 'w')
        train_part.write(schema)
        test_part.write(schema)
        for row in data:
            if (random.randint(0, cv - 1) == i):
                test_part.write(row)
            else:
                train_part.write(row)
        train_part.close()
        test_part.close()


def init_data(data_file, fillna='top', fill_values={}):
    """
    read from file, handling NAN value with value occurred most
    """
    data_set = pandas.load_csv(data_file)

    if (fillna=='top'):
        for i in data_set.columns:
            data_set[i].fillna(data_set[i].describe().top)
            fill_values[i] = data_set[i].describe().top
    else:
        for i in data_set.columns:
            data_set[i].fillna(fill_values[i])
