#!/usr/bin/env python
""" Implements Convolutional Neural Net for training on Zener Cards."""

import sys
import os
import random
import re
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MINI_BATCH_SIZE = 100
LEARNING_RATE = 0.5
EPOCH = 15


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
        self.epsilon = float(epsilon)
        self.max_updates = int(max_updates)
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
                    self.labels.append([1, 0])
                # Check if filename matches negative input class pattern
                elif neg_pattern.match(filename):
                    self.labels.append([0, 1])

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
        # Convert to numpy array
        img_array = np.array(image).astype('uint8')
        # Reshape array to one dimension
        img_array = img_array.reshape(-1)
        # img_array /= 255
        img_array = img_array.tolist()
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

    def train(self, model_file='trained_model', data=None, model_params=None):
        """Trains the Neural Net on data folder

        :param model_file: File to which model is written
        :param data: Input data
        :param model_params: Tensorflow Model parameters
        :returns: Time taken to train
        """

        self.__initialize_variable()

    def __initialize_variable(self):
        """Initializes Tensorflow variable

        :param mode: Mode in which script is run (train,test,k-fold)
        :returns: Train step, input, label, accuracy Tensor depending on mode
        """

        # Input
        x = tf.placeholder(tf.float32, [None, 625], name='x')

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

        print network[-1]

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

        return W, b

    def __get_conv(self, input_x, weight, bias, kernel):
        """Returns a convolutional network layer"""

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

    CONV = CONVNET(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])

    CONV.set_inputs(sys.argv[7])

    CONV.train(sys.argv[2])
