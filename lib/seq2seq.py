import tensorflow as tf
from lib.ops import *

def peeky_seq2seq(
                encoder_inputs,
                decoder_inputs,
                peeky_code,
                word_embedding_matrix,
                vocab_size,
                sequence_length,
                latent_dim,
                peeky_code_dim,
                encoder_length
                ):
        
        

        with tf.variable_scope("encoder") as scope:
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=latent_dim, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=latent_dim, state_is_tuple=True)
            #bi-lstm encoder
            encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=encoder_length,
                inputs=encoder_inputs,
                time_major=False)

            output_fw, output_bw = encoder_outputs
            state_fw, state_bw = state
            encoder_outputs = tf.concat([output_fw,output_bw],2)      #not pretty sure whether to reverse output_bw
            encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
            encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)   



        peeky_code_dup = tf.tile(peeky_code,[1,sequence_length+1])
        peeky_code_dup = tf.reshape(peeky_code_dup,[-1,sequence_length+1,peeky_code_dim])
        decoder_inputs = tf.concat([decoder_inputs,peeky_code_dup],-1)
        decoder_inputs = batch_to_time_major(decoder_inputs,sequence_length+1)


        
        with tf.variable_scope("decoder") as scope:

            r_num = tf.reduce_sum(tf.random_uniform([1], seed=1))
            r_num = r_num
            cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim*2, state_is_tuple=True)

            def test_decoder_loop(prev,i):
                prev_index = tf.stop_gradient(tf.argmax(prev, axis=-1))
                pred_prev = tf.nn.embedding_lookup(word_embedding_matrix, prev_index)
                next_input = tf.concat([pred_prev,peeky_code],-1)
                return next_input


            #the decoder of training
            train_decoder_output,train_decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs = decoder_inputs,
                initial_state = encoder_state,
                attention_states = encoder_outputs,
                cell = cell,
                output_size = vocab_size,
                scope = scope
            )
            
            #the decoder of testing
            scope.reuse_variables()
            test_decoder_output,test_decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs = decoder_inputs,
                initial_state = encoder_state,
                attention_states = encoder_outputs,
                cell = cell,
                output_size = vocab_size,
                loop_function = test_decoder_loop,
                scope = scope
            )   #the test decoder input can be same as train

            test_decoder_logits = tf.stack(test_decoder_output, axis=1)
            test_pred = tf.argmax(test_decoder_logits,axis=-1)

            return train_decoder_output,test_pred
