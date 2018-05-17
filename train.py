from util import *
import json
from model import *

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini')

conlim = config['dim'].getint('para_limit')
queslim = config['dim'].getint('ques_limit')
charlim = config['dim'].getint('char_limit')
hidlim = config['dim'].getint('hidden_layer_size')
encdim = config['dim'].getint('encode_dim')
chardim = config['dim'].getint('char_dim')
num_gpus = config['hp'].getint('NUM_GPUS')

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grads_concat = tf.concat(grads, axis=0)
        grads_mean = tf.reduce_mean(grads_concat, axis=0)

        v = grad_and_vars[0][1]
        average_grads.append((grads_mean, v))
    return average_grads

with open(config['paths']['word_emb']) as f:
    word_emb = np.array(json.load(f), dtype=np.float32)

with open(config['paths']['char_emb']) as f:
    char_emb = np.array(json.load(f), dtype=np.float32)

word_mat = tf.constant(word_emb, dtype=tf.float32)
char_mat = tf.constant(char_emb, dtype=tf.float32)

# load the model
model = QANet(word_emb, char_emb)

# load the TRAIN dataset
parser = get_record_parser()
train_data = get_batch_dataset(config['paths']['TRAIN_REC'], parser)
train_it = train_data.make_one_shot_iterator()
train_init_op = train_it.make_initializer(train_data)

# load the VAL dataset
parser = get_record_parser(is_test=True)
val_data = get_batch_dataset(config['paths']['DEV_REC'], parser)
val_it = val_data.make_one_shot_iterator()
val_init_op = val_it.make_initializer(val_data)
is_training = tf.placeholder(tf.bool)

#build the optimizer
# lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(global_step, tf.float32) + 1))
opt = tf.train.AdamOptimizer(learning_rate=config['hp'].getfloat('LR'),
                             beta1=0.8, beta2=0.999, epsilon=1e-07,
                             use_locking=False)
tower_grads = []
tower_losses = []
for i in range(num_gpus):
    with tf.device('/gpu:%d' % i):
        with tf.name_scope('gpu_%d' % i):
            tf.layers.set_name_reuse(True)
            c, q, ch, qh, y1, y2, qa_id = tf.cond(is_training,
                                      lambda: train_it.get_next(),
                                      lambda: val_it.get_next())

            # forward inference
            p1, p2 = model.forward(c, q, ch, qh)

            # get the loss
            loss = model.get_loss(p1, p2, y1, y2)
            tower_losses.append(loss)

            ## reuse variables for next call
            tf.get_variable_scope().reuse_variables()

            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)

avg_loss = tf.reduce_mean(tower_losses)
grads = average_gradients(tower_grads)
train_op = opt.apply_gradients(grads)

with tf.Session() as sess:
    sess.run([train_init_op, val_init_op])
    sess.run(tf.global_variables_initializer())
    while True:
        try:
            _, loss = sess.run([train_op, avg_loss])
        except tf.errors.OutOfRangeError:
            break
