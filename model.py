import tensorflow as tf
from utils import utils
from lib.ops import *
from lib.seq2seq import *
import numpy as np
import os, sys, jieba
from flags import FLAGS
import csv

class persona_dialogue():

    def __init__(self,args,sess):
        self.sess = sess
        self.word_embedding_dim = 300
        self.drop_rate = 0.1
        self.latent_dim = args.latent_dim
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.printing_step = args.printing_step
        self.saving_step = args.saving_step
        self.num_step = args.num_step
        self.model_dir = args.model_dir
        self.load = args.load
        self.lstm_length = [self.sequence_length+1]*self.batch_size
        self.utils = utils(args)
        self.vocab_size = len(self.utils.word_id_dict)

        self.BOS = self.utils.BOS_id
        self.EOS = self.utils.EOS_id
        self.log_dir = os.path.join(self.model_dir,'log/')
        self.build_graph()

        self.saver = tf.train.Saver(max_to_keep=10)
        self.model_path = os.path.join(self.model_dir,'model')


    def build_graph(self):
        #print('starting building graph')


        with tf.variable_scope("input") as scope:
            self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
            self.train_decoder_sentence = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
            self.train_decoder_targets = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
            self.train_sentiment = tf.placeholder(dtype=tf.float32, shape=(self.batch_size,1))

            BOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.BOS
            EOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.EOS
            train_decoder_targets = tf.concat([self.train_decoder_targets,EOS_slice],axis=1)
            train_decoder_sentence = tf.concat([BOS_slice,self.train_decoder_sentence],axis=1)


        with tf.variable_scope("embedding") as scope:
            init = tf.contrib.layers.xavier_initializer()

            #word embedding
            word_embedding_matrix = tf.get_variable(
                name="word_embedding_matrix",
                shape=[self.vocab_size, self.word_embedding_dim],
                initializer=init,
                trainable = True)

            encoder_inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix, self.encoder_inputs)
            train_decoder_sentence_embedded = tf.nn.embedding_lookup(word_embedding_matrix, train_decoder_sentence)
        

        #seq2seq
        train_decoder_output,test_pred = peeky_seq2seq(
            encoder_inputs=encoder_inputs_embedded,
            decoder_inputs=train_decoder_sentence_embedded,
            peeky_code=self.train_sentiment,
            word_embedding_matrix=word_embedding_matrix,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            latent_dim=self.latent_dim,
            encoder_length=self.lstm_length,
            peeky_code_dim=1
        )
        train_decoder_logits = tf.stack(train_decoder_output, axis=1)
        self.train_pred = tf.argmax(train_decoder_logits,axis=-1)
        self.test_pred = test_pred

        with tf.variable_scope("loss") as scope:
            targets = batch_to_time_major(train_decoder_targets,self.sequence_length+1)
            loss_weights = [tf.ones([self.batch_size],dtype=tf.float32) for _ in range(self.sequence_length+1)]    #the weight at each time step
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                logits = train_decoder_output, 
                targets = targets,
                weights = loss_weights)
            #self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
            tf.summary.scalar('total_loss', self.loss)

        #for v in tf.trainable_variables():
        #    print(v.name,v.get_shape().as_list())


    def train(self):
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        saving_step = self.saving_step
        summary_step = self.printing_step
        cur_loss = 0.0
       
        ckpt = tf.train.get_checkpoint_state(self.model_dir)

        if self.load != '':
            self.saver.restore(self.sess, self.load)
        elif ckpt:
          print('load model from:', self.model_dir)
          saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print('create fresh parameters')
        step = 0
        
        for x,y,s in self.utils.train_data_generator():
            step += 1
            feed_dict = {
                self.encoder_inputs:np.array(x),\
                self.train_decoder_sentence:np.array(y),\
                self.train_decoder_targets:np.array(y),\
                self.train_sentiment:np.array(s).reshape(-1,1)
            }
            _,loss,t_p = self.sess.run([self.train_op, self.loss,self.train_pred],feed_dict)
            cur_loss += loss
            if step%(summary_step)==0:
                print('\n{step}: total_loss: {loss}'.format(step=step,loss=cur_loss/summary_step))
                preds = self.sess.run([self.test_pred],feed_dict)
                d = feed_dict[self.encoder_inputs]
                print('{}\n{} -> {}\n\n'.format(self.utils.id2sent(d[0]), feed_dict[self.train_sentiment][0][0], self.utils.id2sent(preds[0][0])))
                cur_loss = 0.0
            if step%saving_step==0:
                self.saver.save(self.sess, self.model_path, global_step=step)
            if step>=self.num_step:
                break


    def test(self):
        sentence = 'Hi~'
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        jieba.load_userdict('/media/alison/Documents/research/chatbot/_pre/code/segment/dict_fastText_fre.txt')
        print('please enter sentiment from 0 to 1 and sentence')
        print('example: 0.95:i want to leave\n')
        while(sentence):
            sentence = raw_input('>')
            sentiment = float(sentence.split(':')[0])
            sentence = sentence.split(':')[1]
            seg = jieba.cut(sentence, cut_all=False)
            input_sent_vec, _ = self.utils.sent2id(' '.join(seg).encode('utf8'))
            sent_vec = np.zeros((self.batch_size,self.sequence_length),dtype=np.int32)

            sent_vec[0] = input_sent_vec
            sentiment_vec = np.zeros((self.batch_size),dtype=np.float32)
            sentiment_vec[0] = sentiment
            t = np.ones((self.batch_size,self.sequence_length),dtype=np.int32)
            feed_dict = {
                    self.encoder_inputs:sent_vec,\
                    self.train_decoder_sentence:t,
                    self.train_sentiment:sentiment_vec.reshape(-1,1)
            }
            preds = self.sess.run([self.test_pred],feed_dict)
            pred_sent = self.utils.id2sent(preds[0][0])
            print(pred_sent)

    def val(self):
        if self.load != '':
          self.saver.restore(self.sess, self.load)
        else:
          self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

        cf = open(FLAGS.output, 'w')
        writer = csv.writer(cf, delimiter='|')
        writer.writerow(['context', 'utterance'])
           
        for title_vec, title_sen in self.utils.test_data_generator():
            t = np.ones((self.batch_size,self.sequence_length),dtype=np.int32)

            feed_dict = {
                    self.encoder_inputs:np.array(title_vec),\
                    self.train_decoder_sentence:t,
                    self.train_sentiment:np.array([ FLAGS.scale ] * len(title_vec)).reshape(-1,1)
            }

            preds = self.sess.run([self.test_pred], feed_dict)

            for title, p in zip(title_sen, preds[0]):
              title = ''.join(title.split())
              p = self.utils.id2sent(p)
              writer.writerow([title, p])
              #print '{}\n{: <4} -> {}\n1.0 -> {}\n'.format(title, round(s, 2), p, p2)

        cf.close()

