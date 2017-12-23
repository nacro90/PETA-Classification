import tensorflow as tf
import sys

import nn_datasource as ds
from checkpoint_manager import CheckpointManager

NETWORK_NUMBER = 1

IMAGE_WIDTH = 72
IMAGE_HEIGHT = 170
IMAGE_CHANNEL = 3

EPOCH_LENGTH = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.001

DROPOUT = True

RANDOM_SEED = 1

def main():

    global DROPOUT

    input_placeholder = tf.placeholder(
        tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])

    output_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

    layer_conv_1, weights_conv_1 = new_conv_layer(
        input=input_placeholder,
        num_input_channels=IMAGE_CHANNEL,
        filter_size=5,
        num_filters=64,
        pooling=2
    )

    layer_conv_2, weights_conv_2 = new_conv_layer(
        input=layer_conv_1,
        num_input_channels=64,
        filter_size=3,
        num_filters=128,
        pooling=2
    )

    layer_conv_3, weights_conv_3 = new_conv_layer(
        input=layer_conv_2,
        num_input_channels=128,
        filter_size=3,
        num_filters=128,
        pooling=None
    )

    layer_conv_4, weights_conv_4 = new_conv_layer(
        input=layer_conv_3,
        num_input_channels=128,
        filter_size=3,
        num_filters=128,
        pooling=None
    )

    layer_conv_5, weights_conv_5 = new_conv_layer(
        input=layer_conv_4,
        num_input_channels=128,
        filter_size=3,
        num_filters=256,
        pooling=3
    )

    layer_flat, num_features = flatten_layer(layer_conv_5)

    layer_fc_1 = new_fc_layer(
        input=layer_flat, num_inputs=num_features, num_outputs=4096)

    layer_fc_1 = tf.nn.sigmoid(layer_fc_1)

    if DROPOUT is True:
        layer_fc_1 = tf.nn.dropout(layer_fc_1, 0.5)

    layer_fc_2 = new_fc_layer(
        input=layer_fc_1, num_inputs=4096, num_outputs=4096)

    layer_fc_2 = tf.nn.sigmoid(layer_fc_2)

    if DROPOUT is True:
        layer_fc_2 = tf.nn.dropout(layer_fc_2, 0.5)

    layer_output = new_fc_layer(
        input=layer_fc_2, num_inputs=4096, num_outputs=1)

    layer_output = tf.nn.sigmoid(layer_output)

    cost = tf.reduce_sum(tf.pow(layer_output - output_placeholder, 2)) / (2 * BATCH_SIZE)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    correct_prediction = tf.equal(layer_output, output_placeholder)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    checkpoint_manager = CheckpointManager(
        NETWORK_NUMBER, ds.DATASET_FOLDER, EPOCH_LENGTH, BATCH_SIZE, 
        LEARNING_RATE, optimizer.__class__.__name__, DROPOUT)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        checkpoint_manager.on_start()

        for batch_index, training_images, training_labels in ds.training_batch_generator(BATCH_SIZE):

            print("Starting batch {:3}".format(batch_index + 1))

            for current_epoch in range(EPOCH_LENGTH):
                
                DROPOUT = True
                
                feed = {
                    input_placeholder: training_images,
                    output_placeholder: training_labels
                }

                epoch_accuracy, epoch_cost, _ = sess.run(
                    [accuracy, cost, optimizer], feed_dict=feed)
                print("Epoch {:3}".format(current_epoch + 1),
                      "Accuracy: {:6.1%}".format(epoch_accuracy), 
                      "Cost:", epoch_cost)

                checkpoint_manager.on_epoch_completed()

            # Disabled for calculating batch loss
            DROPOUT = False

            batch_accuracy_training, batch_cost_training = sess.run(
                [accuracy, cost], feed_dict=feed)

            print("Batch {} has been finished. ".format(batch_index + 1),
                  "Accuracy: {:6.1%}".format(batch_accuracy_training),
                  "Cost:", batch_cost_training)

            checkpoint_manager.on_batch_completed(
                batch_cost_training, batch_accuracy_training)

            checkpoint_manager.save_model(sess)


        print("\nTraining finished!")

        DROPOUT = False

        overall_accuracy_test, overall_cost_test = sess.run(
            [accuracy, cost], feed_dict=feed)
        print("Accuracy: {:6.1%}".format(overall_accuracy_test), 
              "Cost:", overall_cost_test)

        checkpoint_manager.save_model(sess)

        checkpoint_manager.on_completed(overall_accuracy_test)


def new_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, seed=RANDOM_SEED))


def new_biases(length):
    return tf.Variable(tf.zeros(shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=2):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME')

    layer = tf.add(layer, biases)

    if pooling is not None and pooling > 1:
        layer = tf.nn.max_pool(value=layer, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.add(tf.matmul(input, weights), biases)
    # layer = tf.nn.relu(layer)
    return layer


if __name__ == '__main__':
    main()
