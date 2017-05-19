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
        self.word_id_dict,_ = self.get_dictionary('data/dictionary.txt')
        self.unknown_id = 1
        self.EOS_id = 0
        self.BOS_id = 3

        self.id_word_dict = [[]]*len(self.word_id_dict)
        print(len(self.id_word_dict))
        for word in self.word_id_dict:
            self.id_word_dict[self.word_id_dict[word]] = word

 
    def get_dictionary(self, dict_file):
        if os.path.exists(dict_file):
            print ('loading dictionary from : %s' %(dict_file))
            dictionary = dict()
            num_word = 0
            with open(dict_file, 'r', errors='ignore') as file:
                un_parse = file.readlines()
                for line in un_parse:
                    line = line.strip('\n').split()
                    dictionary[line[0]] = int(line[1])
                    num_word += 1
            return dictionary, num_word
        else:
            raise ValueError('Can not find dictionary file %s' %(dict_file))


    def sent2id(self,sent,l=None):
        pat = re.compile('(\W+)')
        sent_list = ' '.join(re.split(pat,sent.lower().strip())).split()
        vec = np.zeros((self.sequence_length),dtype=np.int32)
        sent_len = len(sent_list)
        unseen = 0
        for i,word in enumerate(sent_list):
            if i==self.sequence_length:
                break
            if word in self.word_id_dict:
                vec[i] = self.word_id_dict[word]
            else:
                unseen += 1
                vec[i] = self.unknown_id
        if unseen>=2:
            sent_len = 0
        if l:
            return vec,sent_len
        else:
            return vec  


    def id2sent(self,ids):
        word_list = []
        for i in ids:
            word_list.append(self.id_word_dict[i])
        return ' '.join(word_list)


    def train_data_generator(self):
        while(True):
            with open('data/open_subtitles_sentiment.txt') as fp:
                batch_x = [];batch_y = [];batch_s = []
                flag = 0
                for line in fp:
                    if flag==0:
                        flag = 1
                        cur_x,cur_x_l = self.sent2id(line.strip().split('==+==')[0],l=True)
                    elif flag == 1:
                        flag = 0
                        cur_y,cur_y_l = self.sent2id(line.strip().split('==+==')[0],l=True)
                        cur_s = float(line.strip().split('==+==')[1])
                        if cur_y_l<self.sequence_length+2 and cur_y_l>2 and cur_x_l<self.sequence_length+2 and cur_x_l>2:
                            batch_x.append(cur_x)
                            batch_y.append(cur_y)
                            batch_s.append(cur_s)
                        if len(batch_x)>=self.batch_size:
                            yield batch_x,batch_y,batch_s
                            batch_x = [];batch_y = [];batch_s = []