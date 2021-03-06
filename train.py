import os
import json
from util import *
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
save_dir = config['paths']['TMPDIR']
log_dir = config['paths']['LOGDIR']

MAX_EPOCH = config['hp'].getint('MAX_EPOCH')
DISPLAY_ITER = config['iter'].getint('DISPLAY_ITER')
SAVE_ITER = config['iter'].getint('SAVE_ITER')
VAL_ITER = config['iter'].getint('VAL_ITER')

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

with open(config['paths']['dev_eval'], "r") as fh:
    dev_eval_f = json.load(fh)

with open(config['paths']['dev_meta'], "r") as fh:
    meta = json.load(fh)

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
# opt = tf.train.GradientDescentOptimizer(0.01)
tower_grads = []
tower_losses = []
for i in range(num_gpus):
    with tf.device('/gpu:%d' % i):
    # with tf.device('/cpu:%d' % i):
        with tf.variable_scope('gpu_%d' % i, reuse=tf.AUTO_REUSE):
            c, q, ch, qh, y1, y2, qa_id = tf.cond(is_training,
                                                  lambda: train_it.get_next(),
                                                  lambda: val_it.get_next())

            # forward inference
            p1_logits, p2_logits = model.forward(c, q, ch, qh)
            p1 = tf.argmax(p1_logits, axis=1)
            p2 = tf.argmax(p2_logits, axis=1)


            # get the loss
            loss = model.get_loss(p1_logits, p2_logits, y1, y2)
            tower_losses.append(loss)

            # loss = tf.Print(loss, [loss], "loss = ")
            ## reuse variables for next call
            # tf.get_variable_scope().reuse_variables()

            grads = opt.compute_gradients(loss)
            # grads = tf.Print(grads, [grads])
            tower_grads.append(grads)

avg_loss = tf.reduce_mean(tower_losses)
grads = average_gradients(tower_grads)
train_op = opt.apply_gradients(grads)


# saver
saver = tf.train.Saver(max_to_keep=3)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt_path = os.path.join(save_dir, 'ckpt')
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

with tf.Session() as sess:
    # initialize the graph
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    tf.logging.set_verbosity(tf.logging.INFO)

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info('Restoring from: %s', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    else:
        tf.logging.info('No checkpoint file found.')

    it = 0
    for epoch in range(MAX_EPOCH):
        sess.run([train_init_op, val_init_op])
        while True:
            try:
                _, loss = sess.run([train_op, avg_loss], feed_dict={is_training: True})
                # loss = sess.run([avg_loss], feed_dict={is_training: True})
                # print(loss)
                # break

                if it % DISPLAY_ITER == 0:
                    tf.logging.info('epoch %d, step %d, loss = %f', epoch, it, loss)
                    loss_summ = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=loss)
                    ])
                    summary_writer.add_summary(loss_summ, it)

                if it % SAVE_ITER == 0 and it > 0:
                    saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)

                if it % VAL_ITER == 0 and it > 0:
                    sess.run(val_init_op)
                    tf.logging.info('validating...')
                    answer_dict = {}
                    losses = []
                    while True:
                        try:
                            _id, loss, yp1, yp2 = sess.run([qa_id, avg_loss, p1, p2], feed_dict={is_training: False})

                            answer_dict_, _ = convert_tokens(
                                dev_eval_f, _id.tolist(), yp1.tolist(), yp2.tolist())
                            answer_dict.update(answer_dict_)
                            losses.append(loss)
                        except tf.errors.OutOfRangeError as e:
                            break

                    loss = np.mean(losses)
                    metrics = evaluate(dev_eval_f, answer_dict)
                    metrics["loss"] = loss
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="val/loss", simple_value=metrics["loss"]), ])
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="val/f1", simple_value=metrics["f1"]), ])
                    em_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="val/em", simple_value=metrics["exact_match"]), ])

                    tf.logging.info('val f1: %f', metrics['f1'])

                    # add val metrics to summary
                    to_write = [loss_sum, f1_sum, em_sum]
                    for metric in to_write:
                        summary_writer.add_summary(metric, it)

            except tf.errors.OutOfRangeError:
                break

            it += 1
        # break
