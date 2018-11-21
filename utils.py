import os
import random
import json
import random
import numpy as np
import re

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


class utils():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.sequence_length = args.sequence_length
        self.set_dictionary(os.path.join(self.data_dir, 'dict'))

    def set_dictionary(self, dict_file):
      if os.path.exists(dict_file):
        fp = open(dict_file,'r')
        self.word_id_dict = json.load(fp)
        #print('word number:',len(self.word_id_dict))

        self.BOS_id = self.word_id_dict['__BOS__']
        self.EOS_id = self.word_id_dict['__EOS__']
        self.UNK_id = self.word_id_dict['__UNK__']

        self.id_word_dict = [[]]*len(self.word_id_dict)
        for word in self.word_id_dict:
            self.id_word_dict[self.word_id_dict[word]] = word
      else:
        print('where is dictionary file QQ?')

    def sent2id(self,sent):
        vec = np.zeros((self.sequence_length),dtype=np.int32) + self.EOS_id

        i, unk = 0, 0
        for word in sent.decode('utf8').split():
            if word in self.word_id_dict:
                vec[i] = self.word_id_dict[word]
            else:
                vec[i] = self.UNK_id
                unk += 1
            i += 1
            if i >= self.sequence_length:
                break
        if unk < 3:
          l = i
          while i < self.sequence_length:
            vec[i] = self.EOS_id
            i += 1
        else:
          l = self.sequence_length + 100

        return vec, l

    def id2sent(self,ids):
        word_list = []
        for i in ids:
            if i == self.EOS_id:
                break
            word_list.append(self.id_word_dict[i])
        if word_list == []:
          word_list = ['.']
        return ''.join(word_list).encode('utf8')


    def train_data_generator(self):
        while(True):
            with open(os.path.join(self.data_dir, 'source_train')) as fp:
                batch_x = []; batch_y = []; batch_s = []
                for line in fp:
                    s, x, y = line.strip().split(' +++$+++ ')
                    s = float(s)
                    x, xl = self.sent2id(x)
                    y, yl = self.sent2id(y)

                    if xl <= self.sequence_length and yl <= self.sequence_length:
                      batch_x.append(x)
                      batch_y.append(y)
                      batch_s.append(s)
                    if len(batch_x) >= self.batch_size:
                      yield batch_x, batch_y, batch_s
                      batch_x = []; batch_y = []; batch_s = []

    def test_data_generator(self):
      with open(os.path.join(self.data_dir, 'source_test')) as fp:
        batch_x = []; batch_xs = []
        for line in fp:
          xs, ys = line.strip().split(' +++$+++ ')
          x, xl = self.sent2id(xs)

          #if xl <= self.sequence_length:
          batch_x.append(x)
          batch_xs.append(xs)
          if len(batch_x) >= self.batch_size:
            yield batch_x, batch_xs
            batch_x = []; batch_xs = []

