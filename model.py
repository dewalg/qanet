"""
QAnet implementation by Dewal Gupta
"""

import tensorflow as tf
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation

_DEBUG = False


class QANet:
    def __init__(self, word_emb, char_emb, is_training=True):
        self.word_emb = tf.constant(word_emb, dtype=tf.float32)
        self.char_emb = tf.constant(char_emb, dtype=tf.float32)
        self.is_training = is_training

        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('config.ini')

        # dimensions and limits for the model
        self.con_lim  = self.config['dim'].getint('para_limit') # if is_training else self.config['dim'].getint('test_para_limit')
        self.ques_lim = self.config['dim'].getint('ques_limit') # if is_training else self.config['dim'].getint('test_ques_limit')
        self.char_lim = self.config['dim'].getint('char_limit')
        self.hid_dim  = self.config['dim'].getint('hidden_layer_size')
        self.enc_dim  = self.config['dim'].getint('encode_dim')
        self.char_dim = self.config['dim'].getint('char_dim')
        self.N        = self.config['dim'].getint('batch_size')

        # hyper params - TODO: put into a config file
        self.emb_kernel_size = 7
        self.emb_num_blocks = 1
        self.emb_num_conv_layers = 4

        self.model_num_blocks = 7
        self.model_num_conv_layers = 2

        self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
        self.dropout = 0.1

        # encoder blocks
        self.emb_encoder = EncoderBlk(self.emb_num_blocks,
                                      self.emb_num_conv_layers,
                                      self.emb_kernel_size,
                                      self.enc_dim,
                                      self.is_training)

        self.model_blk_1 = EncoderBlk(self.model_num_blocks,
                                      self.model_num_conv_layers,
                                      self.emb_kernel_size,
                                      self.enc_dim,
                                      self.is_training)

        self.model_blk_2 = EncoderBlk(self.model_num_blocks,
                                      self.model_num_conv_layers,
                                      self.emb_kernel_size,
                                      self.enc_dim,
                                      self.is_training)

        self.model_blk_3 = EncoderBlk(self.model_num_blocks,
                                      self.model_num_conv_layers,
                                      self.emb_kernel_size,
                                      self.enc_dim,
                                      self.is_training)

    def embed(self, word, char, is_context=False):
        """
        returns the embedding of the word AND char by
        concatenating the two and using highway networks
        :param is_context: boolean value indicating whether we're building
         the embedding for the context or the question
        :param word: the word indices
        :param char: the char indices 
        :return: the combined embedding
        """
        # TODO: add dropout on the embeddings
        emb_dim = self.con_lim if is_context else self.ques_lim
        emb_char = tf.nn.embedding_lookup(self.char_emb, char)
        emb_char = tf.reshape(emb_char, [self.N*emb_dim, self.char_lim, self.char_dim])

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
        emb_char = tf.reshape(emb_char, [self.N, emb_dim, emb_char.shape[-1]])

        emb_word = tf.nn.embedding_lookup(self.word_emb, word)
        # emb_word = tf.reshape(emb_word, [self.N, emb_word.shape[1], emb_word.shape[2]])
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

    def context_query_att(self, c, q):
        """
        computes the context-query attention using the trilinear
        function as described in the qa-net paper. 
        we use different lambda functions to map each 
        question row to each context row, to calculate the similarity.
        *Note we take extra care for the batch dimension
        :param c: the context embedding, a tensor [batch x cLen x D] 
        :param q: the question embedding, a tensor [batch x qLen x D] 
        :return: the similarity matrix, a tensor [batch x cLen x 4*D] 
        """
        W = tf.get_variable("Sim_W_0", [3*self.enc_dim, 1])
        def lambda_map_2(c_row, q_row):
            # 'c' is a tensor looking like [D,]
            # 'q' is a tensor looking like [D,]
            c_row = tf.reshape(c_row, [1, self.enc_dim])
            q_row = tf.reshape(q_row, [1, self.enc_dim])
            cq_mul = tf.multiply(c_row, q_row)
            cqcq = tf.concat([c_row, q_row, cq_mul], axis=1)
            return tf.reshape(tf.matmul(cqcq, W), [])

        def lambda_map_1(c_row, q_single):
            # 'c' is a tensor looking like [D]
            # 'q' is a tensor looking like [qLen x D]
            return tf.map_fn(lambda q_row: lambda_map_2(c_row, q_row), q_single, dtype=tf.float32)

        def lambda_map(c_single, q_single):
            # 'c' is a tensor looking like [cLen x D]
            # 'q' is a tensor looking like [qLen x D]
            return tf.map_fn(lambda c_row: lambda_map_1(c_row, q_single), c_single, dtype=tf.float32)

        # compute S for each batch
        l = []
        for b in range(self.N):
            out = lambda_map(c[b], q[b])
            l.append(out)

        # S should be a matrix of dim [batch x qLen x cLen]
        S = tf.stack(l)

        # do a row-wise softmax
        S_ = tf.nn.softmax(S, axis=1)
        l = []
        for b in range(self.N):
            out = tf.matmul(S_[b], q[b])
            l.append(out)
        A = tf.stack(l)

        # do a row-wise softmax
        S__ = tf.nn.softmax(S_, axis=2)
        l = []
        for b in range(self.N):
            out = tf.matmul(S__[b], S_[b], transpose_b=True)
            out = tf.matmul(out, c[b])
            l.append(out)
        B = tf.stack(l)

        # concatenate the results and return for the next layer
        return tf.concat([c, A, c*A, c*B], axis=2)

    def out(self, model_out_0, model_out_1):
        """
        Outputs a prediction tensor given the two stacked encoder outputs from
        the previous layer. Adapted to work for both start and end pointers. 
        :param model_out_0: output from first block
        :param model_out_1: output from either second or third block, depending
         on whether this is for the start or end pointer. 
        :return: a tensor representing the probabilities of the start/end of the answer 
        """
        W = tf.get_variable("W", [self.N, self.enc_dim * 2, 1])
        inp = tf.concat([model_out_0, model_out_1], axis=2)
        inp = tf.squeeze(tf.matmul(inp, W))

        # scale the tensor so softmax doesn't blow up
        inp = tf.div(
            tf.subtract(inp, tf.reduce_min(inp)),
            tf.subtract(tf.reduce_max(inp), tf.reduce_min(inp))
        )

        logits = tf.nn.softmax(inp)
        p = tf.argmax(logits, axis=1)
        return logits, p

    def forward(self, context, ques, context_char, ques_char):
        # get word and character embeddings
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            c_emb = self.embed(context, context_char, is_context=True)
            q_emb = self.embed(ques, ques_char)

        # encode the embeddings
        with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
            c = self.emb_encoder.forward(c_emb)
            q = self.emb_encoder.forward(q_emb)

        # apply the context-query attention similar to BiDAF model
        with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
            att = self.context_query_att(c, q)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            model_0 = self.model_blk_1.forward(att)
            model_1 = self.model_blk_2.forward(model_0)
            model_2 = self.model_blk_3.forward(model_1)

        with tf.variable_scope("out_p1"):
            p1_logits, p1 = self.out(model_0, model_1)

        with tf.variable_scope("out_p2"):
            p2_logits, p2 = self.out(model_0, model_2)

        return p1_logits, p2_logits

    def get_loss(self, p_logits_1, p_logits_2, actual_1, actual_2):
        p1 = tf.reduce_sum(p_logits_1*actual_1, axis=1)
        p2 = tf.reduce_sum(p_logits_2*actual_2, axis=1)
        p1 = tf.reduce_sum(tf.log(p1))
        p2 = tf.reduce_sum(tf.log(p2))
        return -1 * (p1+p2) / self.N


class EncoderBlk:
    def __init__(self, num_blks, num_conv_layers, kernel_size=7, num_filters=128, is_training=True):
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('config.ini')

        self.num_blks = num_blks
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        self.dim = self.num_filters
        self.is_training = is_training
        self.N = self.config['dim'].getint('batch_size')
        self.dropout_keep_prob = self.config['hp'].getfloat('DROPOUT_KEEP_PROB')

        self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)

        # self attention
        self.attention = Attention(linear_key_dim=self.dim,
                                   linear_value_dim=self.dim,
                                   model_dim=self.dim,
                                   dropout=(1.0 - self.dropout_keep_prob))

    def pos_enc(self, inputs):
        """
        encodes the position into the vector as described by
        Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf)
        :param inputs: the embedded inputs of shape 'batch' by L by D
        :return: output with the positional encoding 
            the output is the same shape as input
        """
        length = inputs.get_shape().as_list()[1]
        encoded_vec = np.array(
            [pos / np.power(10000, 2 * i / self.dim) for pos in range(length) for i in range(self.dim)])
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        encoded_vec = tf.convert_to_tensor(encoded_vec.reshape([length, self.dim]), dtype=tf.float32)
        return inputs + encoded_vec

    def convolve(self, inputs):
        """
        do a separable convolution on the inputs
        :param inputs: a tensor of N x L x C 
        :return: same as input 
        """
        # make inputs into a 4 dimensional vector so
        # we can use tf's sep_conv function
        x = tf.expand_dims(inputs, 2)
        for i in range(self.num_conv_layers):
            residual = x
            x = tf.contrib.layers.layer_norm(x, trainable=True)
            depthwise_filter = tf.get_variable("depthwise_filter",
                                               (self.kernel_size, 1, self.dim, 1),
                                               dtype=tf.float32,
                                               regularizer=self.l2_regularizer)
            pointwise_filter = tf.get_variable("pointwise_filter",
                                               (1, 1, self.dim, self.num_filters),
                                               dtype=tf.float32,
                                               regularizer=self.l2_regularizer)

            x = tf.nn.separable_conv2d(x, depthwise_filter,
                                       pointwise_filter,
                                       strides=(1, 1, 1, 1),
                                       padding="SAME")

            x = tf.add(residual, x)
            if (i+1)%2 == 0:
                x = tf.layers.dropout(x, 1.0-self.dropout_keep_prob, training=self.is_training)

        return tf.squeeze(x, 2)

    def project(self, inputs):
        """
        if our inputs are not in the same dimensional space as our convolutions, 
        then we want to do a simple projection otherwise do nothing
        :param inputs: Tensor to be projected of shape L by d 
        :return: a Tensor of shape L by num_filters 
        """
        dim = inputs.get_shape().as_list()[2]
        if dim == self.dim:
            return inputs

        emb = tf.layers.dense(inputs, self.dim, use_bias=False, name="project")
        return emb

    def self_att(self, inputs):
        inputs_norm = tf.contrib.layers.layer_norm(inputs)
        K = tf.layers.dense(inputs_norm, self.dim)
        V = tf.layers.dense(inputs_norm, self.dim)
        Q = tf.layers.dense(inputs_norm, self.dim)
        out = self.attention.multi_head(Q, K, V)
        return inputs + out

    def feed_forward(self, inputs):
        inputs_norm = tf.contrib.layers.layer_norm(inputs)
        out = tf.layers.dense(inputs_norm, self.num_filters, activation=tf.nn.relu,
                              use_bias=True)
        out = tf.layers.dropout(out, 1.0-self.dropout_keep_prob, training=self.is_training)
        return tf.add(inputs, out)

    def forward(self, inputs):
        inputs = self.project(inputs)
        for _ in range(self.num_blks):
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                out = self.pos_enc(inputs)
                out = self.convolve(out)
                out = self.self_att(out)
                # out = tf.Print(out, [out])
                out = self.feed_forward(out)

        return out


class Attention:
    """
    Attention class for self attention from the 
    paper "All You Need Is Attention" 
    (https://arxiv.org/pdf/1706.03762.pdf)
    
    implementation by @DongjunLee
    https://github.com/DongjunLee/transformer-tensorflow/
    """

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 dropout=0.2):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout

    def multi_head(self, q, k, v):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5)

        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :]) # (batch_size, num_heads, query_dim, key_dim)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (q_dim, k_dim)
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)