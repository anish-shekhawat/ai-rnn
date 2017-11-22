#!/usr/bin/env python
""" Implements Convolutional Neural Net for training on Zener Cards."""

import sys
import os
import random
import re
import time
import numpy as np
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MINI_BATCH_SIZE = 5
LEARNING_RATE = 0.001
EPOCH = 5


class CONVNET(object):
    """A Convolutional Neural Network (NN) that classifies zener card images

    Attirbutes:
        epsilon: Training error
        max_updates: Max no. of updates
        class_letter: Class letter of training model
        pos_input: Positive training samples
        neg_input: Negative training samples
    """

    def __init__(self, cost, epsilon, max_updates, class_letter):
        """ Return a new CONVNET object

        :param cost: Cost function to use
        :param epsilon: Training error
        :param max_updates: No. of maximum updates
        :param class_letter: Image class for which neural net(NN) is trained
        :returns: returns nothing
        """

        self.cost = cost
        global LEARNING_RATE
        global EPOCH
        LEARNING_RATE = float(epsilon)
        EPOCH = int(max_updates)
        self.class_letter = class_letter
        self.input = []
        self.labels = []

    def set_inputs(self, train_folder):
        """Sets positive and negative training data arrays.

        :param train_folder: Folder name where training inputs are stored
        :returns: returns nothing
        """
        # Check if folder exists
        if os.path.isdir(train_folder) is not True:
            print >> sys.stderr, "NO DATA"
            exit(1)

        is_empty = True

        # Training class filename pattern (positive input pattern)
        pos_pattern = re.compile("[0-9]+_" + self.class_letter + ".png")

        class_letter_set = set(list(self.class_letter))

        zener_card_letters = set(['O', 'P', 'Q', 'S', 'W'])

        zener_letters_string = "".join(list(zener_card_letters))

        file_pattern = re.compile("[0-9]+_[" + zener_letters_string + "].png")

        # Get class letters of negative inputs
        neg_class_letter_set = zener_card_letters - class_letter_set

        neg_class_letters = "".join(list(neg_class_letter_set))

        # Non-training class filename pattern (negative input pattern)
        neg_pattern = re.compile("[0-9]+_[" + neg_class_letters + "].png")

        # Convert images to numpy array of pixels
        for filename in os.listdir(train_folder):
            # Get absolute path
            abs_path = os.path.abspath(train_folder) + "/" + filename

            if file_pattern.match(filename):
                is_empty = False
                # Get image array
                img_array = self.__read_image(abs_path)
                # Append to input collection array
                self.input.append(img_array)
                # Check if filename matches training class filename pattern
                if pos_pattern.match(filename):
                    self.labels.append([0, 1])
                # Check if filename matches negative input class pattern
                elif neg_pattern.match(filename):
                    self.labels.append([1, 0])

        # Check if folders are empty
        if is_empty is True:
            print >> sys.stderr, "NO DATA"
            exit(1)

        # print len(self.input)
        # print len(self.labels)

        self.__randomize_inputs(self.input, self.labels)

    def __read_image(self, path):
        """ Returns Numpy array of image pixels

        :param path: Absolute path of the image file
        :returns: Returns numpy array of image
        """

        # Open image using PIL
        image = Image.open(path, 'r')
        # print path

        img_array = np.array(image.getdata()).reshape(
            image.size[0], image.size[1], 1)
        img_array = img_array.tolist()
        # Reshape array to one dimension
        # img_array = img_array.reshape(-1)
        # img_array /= 255
        # img_array = img_array.tolist()
        # print img_array
        # exit()
        return img_array

    def __randomize_inputs(self, data, label):
        """Randomizes the input data

        :param data: Input data
        :param label: Label of input data
        :returns: Returns nothing
        """

        for i in range(0, len(data) - 1):
            j = random.randint(i + 1, len(data) - 1)
            temp = data[i]
            data[i] = data[j]
            data[j] = temp
            temp = label[i]
            label[i] = label[j]
            label[j] = temp

    def cross_validation(self, k, model_file):
        """Performs a k-fold cross validation training and testing

        :params k: k-fold cross validation
        :param model_file: File to which model is written and then read
        :returns: Returns nothing
        """

        x, y, train_step, logits, acc = self.__initialize_variables("train")

        params = {}
        params['accuracy'] = acc
        params['train_step'] = train_step
        params['x'] = x
        params['y'] = y
        params['out_layer'] = logits

        subset_size = len(self.input) / k
        input_subsets = []
        label_subsets = []
        for i in range(0, len(self.input), subset_size):
            subset = self.input[i: i + subset_size]
            input_subsets.append(subset)
            subset = self.labels[i: i + subset_size]
            label_subsets.append(subset)

        accuracy_sum = 0
        train_dur = 0
        test_dur = 0

        for j in range(k):
            input_train_set = []
            label_train_set = []
            input_test_set = []
            label_test_set = []
            for i in range(k):
                if i != j:
                    input_train_set.extend(input_subsets[i])
                    label_train_set.extend(label_subsets[i])
                else:
                    input_test_set = input_subsets[i]
                    label_test_set = label_subsets[i]

            train_dur += self.train(params, input_train_set,
                                    label_train_set, model_file)

            accur, matrix, dur = self.test(
                model_file, input_test_set, label_test_set, params)

            accuracy_sum += accur
            test_dur += dur

        print "Processing complete!"
        print "Total no. of items trained and tested: %s" % len(self.input)
        print "Overall Accuracy over testing data: %s" % (accuracy_sum / k)
        print "Training time: %s seconds" % train_dur
        print "Testing time: %s seconds" % test_dur
        print "Confusion Matrix: "
        print matrix

    def train(self, model_params, data, labels, model_file='trained_model'):
        """Trains the Neural Net on data folder

        :param model_file: File to which model is written
        :param data: Input data
        :param model_params: Tensorflow Model parameters
        :returns: Time taken to train
        """
        start_time = time.time()

        x = model_params['x']
        y = model_params['y']
        train_step = model_params['train_step']

        # Global Initializer
        init = tf.global_variables_initializer()

        # Initialize the TensorFlow session
        sess = tf.InteractiveSession()
        sess.run(init)

        for i in range(EPOCH):
            print "EPOCH: ", i + 1
            counter = 0
            for k in range(0, len(data), MINI_BATCH_SIZE):
                batch_xs = data[k:k + MINI_BATCH_SIZE]
                batch_ys = labels[k:k + MINI_BATCH_SIZE]
                sess.run(train_step, feed_dict={
                    x: batch_xs, y: batch_ys
                })
                counter += len(batch_xs)
                if (counter) % 1000 == 0:
                    print str(counter) + " inputs trained."

        print "Processing complete!!"
        # Save model to file
        saver = tf.train.Saver()
        saver.save(sess, model_file)

        return time.time() - start_time

    def test(self, model_file, data=None, labels=None, params=None):
        """Test the Neural Network"""

        if data is None:
            x, y, accuracy, out_layer = self.__initialize_variables('test')
            data = self.input
            labels = self.labels
        else:
            x = params['x']
            y = params['y']
            out_layer = params['out_layer']
            accuracy = params['accuracy']

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        print ("Model restored!")

        start_time = time.time()

        test_accuracy = accuracy.eval(session=sess, feed_dict={
            x: data, y: labels})

        prediction = tf.argmax(out_layer, 1)
        actual = tf.argmax(labels, 1)
        pred, act = sess.run([prediction, actual], feed_dict={
            x: data, y: labels})

        confusion_matrix = tf.contrib.metrics.confusion_matrix(
            act, pred).eval(session=sess)

        duration = time.time() - start_time

        if params is None:
            print "Total number of items tested on: %s" % len(data)
            print "Overall Accuracy over testing data: %s" % test_accuracy
            print "Testing time: %s seconds" % duration
            print "Confusion Matrix: "
            print confusion_matrix
        else:
            print "Number of items tested on: %s" % len(data)
            print "Accuracy over testing data: %s" % test_accuracy

        return test_accuracy, confusion_matrix, duration

    def __initialize_variables(self, mode):
        """Initializes Tensorflow variable

        :param mode: Mode in which script is run (train,test)
        :returns: Train step, input, label, accuracy Tensor depending on mode
        """

        # Input
        x = tf.placeholder(tf.float32, [None, 25, 25, 1], name="x")

        # Label
        y = tf.placeholder(tf.float32, [None, 2], name="y")

        network = self.__get_network_details()

        # Dictionary to store weights and biases
        W = {}
        b = {}
        prev_dim = 1
        out_layer = x

        for i in range(len(network) - 1):
            W[i], b[i] = self.__get_wb(i, network[i][0], prev_dim,
                                       network[i][1], 'conv2d')
            out_layer = self.__get_conv(out_layer, W[i], b[i], network[i][0])
            prev_dim = network[i][1]

        i = len(network) - 1

        reshape_val = out_layer.get_shape().as_list()[1]
        # Flatten max_pool
        out_layer = tf.reshape(
            out_layer, [-1, reshape_val * reshape_val * prev_dim])

        W[i], b[i] = self.__get_wb(i, reshape_val, prev_dim,
                                   network[i][0], 'dense')
        out_layer = self.__get_dense(out_layer, W[i], b[i], 'relu')
        prev_dim = network[i][0]

        i = i + 1

        # Output layer with sigmoid activation
        W[i], b[i] = self.__get_wb(i, 1, prev_dim, 2, 'dense')
        out_layer = self.__get_dense(out_layer, W[i], b[i], '')

        reg_dict = {
            'cross': tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(0.0),  W.values()),
            "cross-l1": tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(0.01), W.values()),
            "cross-l2":  tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(0.01), W.values()),
            "ctest": 0
        }

        reg = reg_dict[self.cost]

        cost = tf.nn.softmax_cross_entropy_with_logits(
            logits=out_layer, labels=y) + reg

        out_layer = tf.nn.softmax(out_layer)

        prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32),
                                  name='accuracy')

        train_step = tf.train.GradientDescentOptimizer(
            LEARNING_RATE).minimize(cost)

        if mode == "train":
            return x, y, train_step, out_layer, accuracy
        elif mode == "test":
            return x, y, accuracy, out_layer

    def __get_network_details(self):
        """Fetches Network details from Network Description file

        :returns: List of network details
        """
        out = []

        with open(sys.argv[2]) as net_file:
            for line in net_file.readlines():
                out.append(map(int, line.split()))

        return out

    def __get_wb(self, index, kernel, prev_dim, features, layer):
        """Returns Weight and Bias TF variables

        """
        weight = 'W' + str(index)
        bias = 'b' + str(index)
        initializer = tf.contrib.layers.xavier_initializer()

        if layer == 'conv2d':
            shape = [kernel, kernel, prev_dim, features]
        else:
            shape = [kernel * kernel * prev_dim, features]

        W = tf.get_variable(
            weight, shape, dtype=tf.float32, initializer=initializer)

        b = tf.get_variable(bias, [features], dtype=tf.float32,
                            initializer=initializer)

        # print W
        # print
        return W, b

    def __get_conv(self, input_x, weight, bias, kernel):
        """Returns a convolutional network layer"""
        print weight
        layer = tf.nn.conv2d(input_x, weight, strides=[1, kernel, kernel, 1],
                             padding='SAME')
        layer = tf.nn.bias_add(layer, bias)
        layer = tf.nn.relu(layer)

        size = [1, 2, 2, 1]
        return tf.nn.max_pool(layer, ksize=size, strides=size, padding='SAME')

    def __get_dense(self, X, weights, bias, activation):
        """Returns a dense layer"""

        node = tf.add(tf.matmul(X, weights), bias)

        if activation == 'relu':
            return tf.nn.relu(node)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(node)

        return node


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 8:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python conv_train.py cost network_description",
        print >> sys.stderr, "epsilon max_updates class_letter",
        print >> sys.stderr, "model_file_name train_folder_name\""
        exit(1)

    if sys.argv[1] not in ['cross', 'cross-l1', 'cross-l2', 'ctest']:
        print >> sys.stderr, "Cost must be one of the following: ",
        print >> sys.stderr, "cross, cross-l1, cross-l2 or ctest"
        exit(1)
    # Check if data folder exists
    if os.path.isdir(sys.argv[7]) is not True:
        print >> sys.stderr, "No folder name " + sys.argv[3] + " found."
        print >> sys.stderr, "Please enter a valid data folder name"
        exit(1)

    CONV = CONVNET(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])

    CONV.set_inputs(sys.argv[7])

    if sys.argv[1] == 'ctest':
        CONVV.test(sys.argv[6])
    else:
        CONV.cross_validation(5, sys.argv[6])
