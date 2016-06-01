#!/bin/bash

label_col_name=$1

rm -rf split/
mkdir split

head -1 ffm_features > ffm_head
split -l 200000 ffm_features split/ffm_features.

for i in $(ls split/ffm_features.*); do
    python3 /home/huang_anli/kaggle_tools/ffm_trans.py $i ffm_head ${label_col_name} > $i.res &
done
wait


cat split/ffm_features.*.res > ffm_features.transed
