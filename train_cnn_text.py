import TextCNN
import data_preprocess
import numpy as np
import tensorflow as tf
import time
import os
import datetime

# global variables
SEQUENCE_LENGTH = 17
EMBEDDING_DIM = 100
CONTROL_FILE = 'control_origin.txt'
DEMENTIA_FILE = 'dementia_origin.txt'
W2V_MODEL = '100features_20context_20mincount_zht'

# data preprocessing, load model, w2v_lookup_table, train_x, train_y, train_x_seg, train_x_onehot
w2v_model = data_preprocess.load_wordvec_model(W2V_MODEL)
x, y = data_preprocess.read_sentence(DEMENTIA_FILE, CONTROL_FILE)
x_seg = data_preprocess.segmentation(x)
x_onehot, vocab_processor = data_preprocess.text_to_onehot(x_seg)
# Split data into train and validate part
x_train, x_dev, y_train, y_dev = data_preprocess.cross_validate_data(
    x_onehot, y)
del x_onehot, y

# hyper parameters
batch_size = 32
num_epochs = 10
dropout_keep_prob = 0.5

num_checkpoints = 5
checkpoint_every = 100
evaluate_every = 100

filter_sizes = (3, 4, 5)
num_filters = 128
l2_reg_lambda = 0.2
SEQUENCE_LENGTH = 17
EMBEDDING_DIM = 100

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN.TextCNN(
            sequence_length=SEQUENCE_LENGTH,
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=EMBEDDING_DIM,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        timestamp = datetime.datetime.now().isoformat()
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir, "runs_2", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}".format(
                time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}".format(
                time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_preprocess.batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
