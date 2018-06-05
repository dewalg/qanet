import os
import json
from util import *
from extra import *
from model import *
from comet_ml import Experiment

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
SUM_ITER = 250

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

experiment = Experiment(api_key="CwfC44eKOZx1oFdva2nDp3P8i", project_name="nlp")
hyper_params = {"epochs": MAX_EPOCH, "learning_rate": config['hp'].getfloat('LR'), "batch_size": config['dim'].getint('batch_size')}
experiment.log_multiple_params(hyper_params)

#build the optimizer
# lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(global_step, tf.float32) + 1))
opt = tf.train.AdamOptimizer(learning_rate=config['hp'].getfloat('LR'),
                             beta1=0.8, beta2=0.999, epsilon=1e-07,
                             use_locking=False)
# load the model

with tf.device('/gpu:0'):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        c, q, ch, qh, y1, y2, qa_id = tf.cond(is_training,
                                              lambda: train_it.get_next(),
                                              lambda: val_it.get_next())

        # forward inference
        model = QANet(word_emb, char_emb)
        p1_logits, p2_logits = model.forward(c, q, ch, qh)
        c_mask = tf.cast(c, tf.bool)
        p1_logits = mask_logits(p1_logits, c_mask)
        p2_logits = mask_logits(p2_logits, c_mask)

        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(p1_logits), axis=2),
                          tf.expand_dims(tf.nn.softmax(p2_logits), axis=1))
        outer = tf.matrix_band_part(outer, 0, config['dim'].getint("ans_limit"))
        p1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        p2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        # get the loss
        train_loss = model.get_loss(p1_logits, p2_logits, y1, y2)

        ## reuse variables for next call
        # tf.get_variable_scope().reuse_variables()
        grads = opt.compute_gradients(train_loss)

# avg_loss = tf.reduce_mean(tower_losses)
# grads = average_gradients(tower_grads)

for grad, var in grads:
    if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

train_op = opt.apply_gradients(grads)
for var in tf.trainable_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))

summary_op = tf.summary.merge(summaries)

# saver
saver = tf.train.Saver(max_to_keep=3)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt_path = os.path.join(save_dir, 'ckpt')
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # initialize the graph
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter("./log", sess.graph)
    tf.logging.set_verbosity(tf.logging.INFO)

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info('Restoring from: %s', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    else:
        tf.logging.info('No checkpoint file found.')

    it = 0
    for epoch in range(MAX_EPOCH):
        experiment.log_current_epoch(epoch)
        sess.run([train_init_op, val_init_op])
        while True:
            try:
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                _, loss, p_1 = sess.run([train_op, train_loss, p1], feed_dict={is_training: True})
                experiment.log_metric("loss", loss, step=it)
                # summary_writer.add_run_metadata(run_metadata, 'step001')
                it += 1

                if it % DISPLAY_ITER:
                    tf.logging.info('epoch %d, step %d, loss = %f', epoch, it, loss)
                    # tf.logging.info('step %d, loss = %f', it, loss)
                    loss_summ = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=loss)
                    ])
                    summary_writer.add_summary(loss_summ, it)

                if it % SAVE_ITER == 0 and it > 0:
                    saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)

                if it % SUM_ITER:
                    summary_str = sess.run(summary_op, feed_dict={is_training: True})
                    summary_writer.add_summary(summary_str, it)

                if it % VAL_ITER == 0 and it > 0:
                    sess.run(val_init_op)
                    tf.logging.info('validating...')
                    answer_dict = {}
                    losses = []
                    step = 0
                    while step < 157:
                        try:
                            _id, vloss, yp1, yp2 = sess.run([qa_id, train_loss, p1, p2], feed_dict={is_training: False})

                            tf.logging.info('step %d, val_loss = %f', step, vloss)
                            answer_dict_, _ = convert_tokens(
                                dev_eval_f, _id.tolist(), yp1.tolist(), yp2.tolist())
                            answer_dict.update(answer_dict_)
                            losses.append(loss)
                            step += 1
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

                    tf.logging.info('val loss: %f', metrics['loss'])
                    tf.logging.info('val f1: %f', metrics['f1'])
                    tf.logging.info('val em: %f', metrics['exact_match'])

                    with experiment.validate():
                        experiment.log_metric("val loss", metrics['loss'], step=it)
                        experiment.log_metric("val f1", metrics['f1'], step=it)
                        experiment.log_metric("val em", metrics['exact_match'], step=it)

                    # add val metrics to summary
                    to_write = [loss_sum, f1_sum, em_sum]
                    for metric in to_write:
                        summary_writer.add_summary(metric, it)

            except tf.errors.OutOfRangeError as e:
                break

        experiment.log_epoch_end(epoch, it)
    summary_writer.close()
