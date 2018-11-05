import argparse
import tensorflow as tf
from model import persona_dialogue
from flags import FLAGS

def run():
    sess = tf.Session()
    model = persona_dialogue(FLAGS, sess)
    if FLAGS.mode == 'train':
        model.train()
    if FLAGS.mode == 'test':
        model.test()
    if FLAGS.mode == 'val':
        model.val()

if __name__ == '__main__':
    run()

