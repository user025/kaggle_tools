
# coding: utf-8

# In[4]:
import sys

def gen_output(raw_data, head_data, outcome, target):
    raw_data = open(sys.argv[1], 'r').readlines()
    field_num = 0
    fields = {}
    col = open(head_data, 'r').readline().rstrip().split(',')

    for i in col:
        if i.find('__') > 0:
            key = i.split('__')[0]
        else:
            key = i
        if key in fields:
            continue
        else:
            field_num += 1
            fields[key] = field_num

    for row in raw_data:
        sample = []
        each_row = row.rstrip('\r\n').split(',')

        if (each_row[-1] == outcome):
            continue

        if (each_row[-1] == target):
            label = '1'
        else:
            label = '0'

        sample.append(str(label))
        for i,p in zip(col, range(len(col))):
            if (i == outcome):
                continue

            val = each_row[p]
            if (i.find('__')>0):
                fe_num = fields[i.split('__')[0]]
            else:
                fe_num = fields[i]
            sample.append('{}:{}:{}'.format(fe_num, p+1, val))

        print(' '.join(sample))

# In[ ]:
if __name__=='__main__':
    if (len(sys.argv) > 3):
        target = sys.argv[4]
    else:
        target = '1'
    gen_output(sys.argv[1], sys.argv[2], sys.argv[3], target)
