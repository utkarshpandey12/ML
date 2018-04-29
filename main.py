import pandas as pd
import numpy as np
import tensorflow as tf
feature = pd.read_csv('combined_csv.csv',header = None)
feature= feature.values
label = pd.read_csv('trlabel.csv',header = None)
label=label.values
testfeature = pd.read_csv('Book1.csv',header = None)
testfeature=testfeature.values
testlabel = pd.read_csv('label.csv',header = None)
testlabel=testlabel.values
training_digits_pl = tf.placeholder("float", [None, 784])
test_digit_pl = tf.placeholder("float", [784])
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))
distance = tf.reduce_sum(l1_distance, axis=1)
pred = tf.argmin(distance, 0)
accuracy = 0.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(testfeature)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={training_digits_pl: feature, test_digit_pl: testfeature[i, :]})

        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(label[nn_index]),  "True Label:",  np.argmax(testlabel[i]))
        # Calculate accuracy
        if np.argmax(label[nn_index]) == np.argmax(testlabel[i]):
            accuracy += 1./len(testfeature)

    print("Done!")
    print("Accuracy:", accuracy)
