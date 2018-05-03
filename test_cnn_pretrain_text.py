# import Modules
import TextCNN_w2v_pretrain
import data_preprocess
import numpy as np
import tensorflow as tf
import time
import os
import datetime
import data_preprocess
import csv

# global variables
SEQUENCE_LENGTH = 17
EMBEDDING_DIM = 100
W2V_MODEL = '100features_20context_20mincount_zht'
CONTROL_FILE = 'control_test.txt'
DEMENTIA_FILE = 'dementia_test.txt'
batch_size = 1
#data preprocessing, load model, w2v_lookup_table, train_x, train_y, train_x_seg, train_x_onehot
w2v_model, word_embedding = data_preprocess.load_wordvec_model(W2V_MODEL)
x, y_test = data_preprocess.read_sentence(DEMENTIA_FILE, CONTROL_FILE)
x_seg = data_preprocess.segmentation(x)
x_test, vocab_processor = data_preprocess.text_to_onehot_w2v(x_seg, w2v_model.wv.vocab.keys())
checkpoint_dir = "/home/yyliu/code/NLP/src/runs_2/1525247189/checkpoints"
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_preprocess.batch_iter(list(x_test), 2, batch_size, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

print(all_predictions)
# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = 0.0
    for i in range(len(all_predictions)):
        if all_predictions[i]==0 and y_test[i][0]==1 or all_predictions[i]==1 and y_test[i][1]==1:
            correct_predictions+=1
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_seg), all_predictions))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)