"""
QAnet implementation by Dewal Gupta
"""

import tensorflow as tf
from configparser import ConfigParser, ExtendedInterpolation

class QANet:
    def __init__(self, word_emb, char_emb):
        self.word_emb = tf.constant(word_emb, dtype=tf.float32)
        self.char_emb = tf.constant(char_emb, dtype=tf.float32)

        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('config.ini')

        # dimensions and limits for the model
        self.con_lim  = self.config['dim'].getint('para_limit')
        self.ques_lim = self.config['dim'].getint('ques_limit')
        self.char_lim = self.config['dim'].getint('char_limit')
        self.hid_dim  = self.config['dim'].getint('hidden_layer_size')
        self.char_dim = self.config['dim'].getint('char_dim')

        # hyper params - TODO: put into a config file
        self.num_embed_blocks = 3
        self.num_out_blocks = 7

        self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
        self.dropout = 0.1

    def embed(self, word, char, isContext=False):
        """
        returns the embedding of the word AND char by
        concatenating the two and using highway networks
        :param isContext: boolean value indicating whether we're building
         the embedding for the context or the question
        :param word: the word indices
        :param char: the char indices 
        :return: the combined embedding
        """
        # TODO: add dropout on the embeddings
        emb_dim = self.con_lim if isContext else self.ques_lim
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            emb_char = tf.nn.embedding_lookup(self.char_emb, char)
            emb_char = tf.reshape(emb_char, [emb_dim, self.char_lim, self.char_dim])

            # as per Seo et al., we do a 1 layer, 1D convolution
            # and then find the max per row, and use that 1D vector
            # as the character embedding
            # use kernel size of 5
            kernel = tf.get_variable("emb_kernel", [5, self.char_dim, self.hid_dim],
                                     dtype=tf.float32, regularizer=self.l2_regularizer)

            bias = tf.get_variable("emb_bias", [1, 1, self.hid_dim],
                                   regularizer=self.l2_regularizer,
                                   initializer=tf.zeros_initializer())

            outputs = tf.nn.conv1d(emb_char, kernel, 1, "VALID") + bias
            outputs = tf.nn.relu(outputs)
            emb_char = tf.reduce_max(outputs, axis=1)
            emb_char = tf.reshape(emb_char, [1, emb_dim, emb_char.shape[-1]])

            emb_word = tf.nn.embedding_lookup(self.word_emb, word)
            emb = tf.concat([emb_word, emb_char], axis=2)

            # 2 layer highway network
            with tf.variable_scope("highway", reuse=tf.AUTO_REUSE):
                # initially project the vector to the same space
                emb = tf.layers.dense(emb, self.hid_dim, use_bias=False, name="init_proj")
                for i in range(2):
                    T = tf.layers.dense(emb, self.hid_dim, activation=tf.sigmoid,
                                        use_bias=True, name="gate_%d" % i)
                    H = tf.layers.dense(emb, self.hid_dim, activation=tf.nn.relu,
                                        use_bias=True, name="affine_%d" % i)
                    H = tf.nn.dropout(H, 1.0 - self.dropout)
                    emb = T*H + emb * (1.0 - T)

            return emb

    def forward(self, context, ques, context_char, ques_char):
        # get word and character embeddings
        c_emb = self.embed(context, context_char, isContext=True)
        q_emb = self.embed(ques, ques_char)

        # encode the embeddings
        c = self.encode_context(c_emb)
        q = self.encode_ques(q_emb)

        att = self.self_attention(c, q)

        blk1_out = att
        for _ in range(7):
            blk1_out = self.model_block_0(blk1_out)

        blk2_out = blk1_out
        for _ in range(7):
            blk2_out = self.model_block_0(blk2_out)

        blk3_out = blk2_out
        for _ in range(7):
            blk3_out = self.model_block_0(blk3_out)

        p_start, p_end = self.out(blk1_out, blk2_out, blk3_out)


    def get_loss(self, pred_1, pred_2, actual_1, actual_2):
        pass
