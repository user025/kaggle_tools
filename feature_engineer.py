#!/usr/bin/env python3

from collections import OrderedDict

class FeatureEncoder(object):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.feature_dict = OrderedDict()

    def fit(self, raw_data):
        """
        this mothod should be overridden
        """
        pass

    def transform(self, raw_data):
        """
        this mothod should be overridden
        """
        return raw_data.copy()

    def fit_transform(self, raw_data):
        self.fit(raw_data)
        return self.transform(raw_data)


def one_hot(feature_dict, key, raw_data, threshold=15):
    counts = raw_data.value_counts()
    for fe, count in counts.items():
        if (count > threshold):
            new_key = key + '__'  + fe
            feature_dict[new_key] = lambda x, i:[int(j == i) for j in x]
        else:
            pass
