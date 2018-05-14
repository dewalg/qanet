"""
QAnet implementation by Dewal Gupta
"""

import tensorflow as tf
import numpy as np

class qanet:
    def __init__(self, word_emb, char_emb):
        self.word_emb = tf.constant(word_emb, dtype=tf.float32)
        self.char_emb = tf.constant(char_emb, dtype=tf.float32)

        # hyper params - TODO: put into a config file
        self.num_embed_blocks = 3
        self.num_out_blocks = 7

    def run(self, context, question):



    def get_loss(self, pred_1, pred_2, actual_1, actual_2):
        pass
