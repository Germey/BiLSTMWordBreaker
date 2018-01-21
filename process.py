import re
from itertools import chain
import pandas as pd
import numpy as np
import pickle

# Read origin data
text = open('data/data.txt', encoding='utf-8').read()
# Get split sentences
sentences = re.split('[，。！？、‘’“”]/[bems]', text)
# Filter sentences whose length is 0
sentences = list(filter(lambda x: x.strip(), sentences))
# Strip sentences
sentences = list(map(lambda x: x.strip(), sentences))

# To numpy array
words, labels = [], []
print('Start creating words and labels...')
for sentence in sentences:
    groups = re.findall('(.)/(.)', sentence)
    arrays = np.asarray(groups)
    words.append(arrays[:, 0])
    labels.append(arrays[:, 1])
print('Words Length', len(words), 'Labels Length', len(labels))
print('Words Example', words[0])
print('Labels Example', labels[0])

# Merge all words
all_words = list(chain(*words))
# All words to Series
all_words_sr = pd.Series(all_words)
# Get value count, index changed to set
all_words_counts = all_words_sr.value_counts()
# Get words set
all_words_set = all_words_counts.index
# Get words ids
all_words_ids = range(1, len(all_words_set) + 1)

# Dict to transform
word2id = pd.Series(all_words_ids, index=all_words_set)
id2word = pd.Series(all_words_set, index=all_words_ids)

# Tag set and ids
tags_set = ['x', 's', 'b', 'm', 'e']
tags_ids = range(len(tags_set))

# Dict to transform
tag2id = pd.Series(tags_ids, index=tags_set)
id2tag = pd.Series(tags_set, index=tag2id)

max_length = 32

def x_transform(words):
    ids = list(word2id[words])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


def y_transform(tags):
    ids = list(tag2id[tags])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


print('Starting transform...')

data_x = list(map(lambda x: x_transform(x), words))
data_y = list(map(lambda y: y_transform(y), labels))

print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
print('Data X Example', data_x[0])
print('Data Y Example', data_y[0])

data_x = np.asarray(data_x)
data_y = np.asarray(data_y)

from os import makedirs
from os.path import exists, join

path = 'data/'

if not exists(path):
    makedirs(path)

print('Starting pickle to file...')
with open(join(path, 'data.pkl'), 'wb') as f:
    pickle.dump(data_x, f)
    pickle.dump(data_y, f)
    pickle.dump(word2id, f)
    pickle.dump(id2word, f)
    pickle.dump(tag2id, f)
    pickle.dump(id2tag, f)
print('Pickle finished')
