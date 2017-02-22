#-*- coding:utf-8 -*-




def set2dict(value_set):
  li = list(value_set)
  return dict((v, k) for k, v in enumerate(li))

def bucketEncode(column_value,boundaries=[]):
    if column_value < boundaries[0]: return 0
    for i in range(1,len(boundaries) - 1):
        if column_value >= boundaries[i] and column_value < boundaries[i+1]:
            return i
    if column_value >= boundaries[len(boundaries) - 1]:
        return len(boundaries) - 1
    return 0