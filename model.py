import tensorflow as tf
from utils import utils
from lib.ops import *
from lib.seq2seq import *
import numpy as np
import os
import sys

class persona_dialogue():

    def __init__(self,args,sess):
        self.sess = sess
        self.word_embedding_dim = 300
        self.drop_rate = 0.1
        self.num_epochs = 10000
        self.num_steps = args.num_steps
        self.latent_dim = args.latent_dim
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.saving_step = args.saving_step
        self.model_dir = args.model_dir
        self.load_model = args.load
        self.lstm_length = [self.sequence_length+1]*self.batch_size
        self.utils = utils(args)
        self.vocab_size = len(self.utils.word_id_dict)

        self.EOS = self.utils.EOS_id
        self.BOS = self.utils.BOS_id
        print(self.BOS)
        self.log_dir = os.path.join(self.model_dir,'log/')
        self.build_graph()

        self.saver = tf.train.Saver(max_to_keep=2)
        self.model_path = os.path.join(self.model_dir,'model_{m_type}'.format(m_type='peeky'))


    def build_graph(self):
        print('starting building graph')


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

        for v in tf.trainable_variables():
            print(v.name,v.get_shape().as_list())


    def train(self):
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        saving_step = self.saving_step
        summary_step = saving_step/10
        cur_loss = 0.0
        
        if self.load_model:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
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
                print('{step}: total_loss: {loss}'.format(step=step,loss=cur_loss/summary_step))
                cur_loss = 0.0
                self.test_when_train(feed_dict)
            if step%saving_step==0:
                self.saver.save(self.sess, self.model_path, global_step=step)
            if step>=self.num_steps:
                break


    def stdin_test(self):
        sentence = 'hi'
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        print('please enter sentiment from 0 to 1 and sentence')
        print('example:0.9:i love you')
        while(sentence):
            sentence = input('>')
            sentiment = float(sentence.split(':')[0])
            sentence = sentence.split(':')[1]
            input_sent_vec = self.utils.sent2id(sentence)
            print(input_sent_vec)
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


    def test_when_train(self,feed_dict):
        train_result_fp = open('train_result.txt','a')
        preds = self.sess.run([self.test_pred],feed_dict)
        d = feed_dict[self.encoder_inputs]
        i = 0
        for one_pred,one_d in zip(preds[0],d):
            train_result_fp.write(self.utils.id2sent(one_d) + '\n')
            train_result_fp.write(self.utils.id2sent(one_pred) + ' ' + str(feed_dict[self.train_sentiment][i]) + '\n\n\n')
            i += 1