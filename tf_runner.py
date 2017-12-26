import sys
import tensorflow as tf

import nn_datasource as ds
from checkpoint_manager import CheckpointManager

IMAGE_WIDTH = 72
IMAGE_HEIGHT = 170
IMAGE_CHANNEL = 3

EPOCH_LENGTH = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.001

RANDOM_SEED = 2

WEIGHT_COUNTER = 0
BIAS_COUNTER = 0
CONVOLUTION_COUNTER = 0
POOLING_COUNTER = 0


def main():

    tf.reset_default_graph()

    TEST = None
    NETWORK_NUMBER = None

    if len(sys.argv) > 2 and sys.argv[1] is not None and sys.argv[1] is not None:
        if sys.argv[1] == 'test':
            TEST = True
        elif sys.argv[1] == 'train':
            TEST = False
        else:
            raise ValueError("Invalid command line argument")

        NETWORK_NUMBER = int(sys.argv[2])
    else:
        raise ValueError("Enter a command line argument [test/train]")

    print(NETWORK_NUMBER)

    input_placeholder = tf.placeholder(
        tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name='input_placeholder')

    output_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='output_placeholder')

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

    if TEST is not True:
        layer_fc_1 = tf.nn.dropout(layer_fc_1, 0.5)

    layer_fc_2 = new_fc_layer(
        input=layer_fc_1, num_inputs=4096, num_outputs=4096)

    layer_fc_2 = tf.nn.sigmoid(layer_fc_2)

    if TEST is not True:
        layer_fc_2 = tf.nn.dropout(layer_fc_2, 0.5)

    layer_output = new_fc_layer(
        input=layer_fc_2, num_inputs=4096, num_outputs=1)

    layer_output = tf.nn.sigmoid(layer_output)

    cost = tf.reduce_sum(
        tf.pow(layer_output - output_placeholder, 2)) / (2 * BATCH_SIZE)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    correct_predictions = tf.equal(layer_output, output_placeholder)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    if TEST is False:
        train_nn(NETWORK_NUMBER, input_placeholder, output_placeholder, accuracy, cost, optimizer)
    elif TEST is True:
        test_nn(NETWORK_NUMBER, input_placeholder, output_placeholder, accuracy, cost, limit=1000)
    else:
        raise ValueError("Invalid TEST value!")


def train_nn(number, input_placeholder, output_placeholder, accuracy, cost, optimizer):
    checkpoint_manager = CheckpointManager(number)
    
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)

        checkpoint_manager.on_training_start(
            ds.DATASET_FOLDER, EPOCH_LENGTH, BATCH_SIZE,
            LEARNING_RATE, "AdamOptimizer", True)

        for batch_index, batch_images, batch_labels in ds.training_batch_generator(BATCH_SIZE):

            print("Starting batch {:3}".format(batch_index + 1))

            for current_epoch in range(EPOCH_LENGTH):

                feed = {
                    input_placeholder: batch_images,
                    output_placeholder: batch_labels
                }

                epoch_accuracy, epoch_cost, _ = sess.run(
                    [accuracy, cost, optimizer], feed_dict=feed)
                print("Batch {:3}, Epoch {:3} -> Accuracy: {:3.1%}, Cost: {}".format(
                    batch_index + 1, current_epoch + 1, epoch_accuracy, epoch_cost))

                checkpoint_manager.on_epoch_completed()

            batch_accuracy_training, batch_cost_training = sess.run(
                [accuracy, cost], feed_dict=feed)

            print("Batch {} has been finished. Accuracy: {:3.1%}, Cost: {}".format(
                batch_index + 1, batch_accuracy_training, batch_cost_training))

            checkpoint_manager.on_batch_completed(
                batch_cost_training, batch_accuracy_training)

            checkpoint_manager.save_model(sess)

        print("\nTraining finished!")

        overall_accuracy, overall_cost = \
            test_nn(number, input_placeholder, output_placeholder, accuracy, cost, limit=None)

        checkpoint_manager.on_training_completed(overall_accuracy)
        

def test_nn(number, input_placeholder, output_placeholder, accuracy, cost, limit=None):
    checkpoint_manager = CheckpointManager(number)


    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)
        checkpoint_manager.restore_model(sess)

        counter = 0
        total_accuracy = 0
        total_cost = 0
        for test_images, test_labels in ds.test_batch_generator(BATCH_SIZE):

            feed = {
                input_placeholder: test_images,
                output_placeholder: test_labels
            }

            test_accuracy, test_cost = sess.run(
                [accuracy, cost], feed_dict=feed)
            print("Batch {:3}, Accuracy: {:3.1%}, Cost: {}" \
                  .format(counter, test_accuracy, test_cost))

            total_accuracy += test_accuracy
            total_cost += test_cost
            counter += 1

        overall_accuracy = total_accuracy / counter
        overall_cost = total_cost / counter

        print("Total test accuracy: {:5.1%}".format(overall_accuracy))

        return overall_accuracy, overall_cost


def new_weights(shape):
    global WEIGHT_COUNTER
    weight = tf.Variable(tf.random_normal(
        shape=shape, seed=RANDOM_SEED), name='w_' + str(WEIGHT_COUNTER))
    WEIGHT_COUNTER += 1
    return weight


def new_biases(length):
    global BIAS_COUNTER
    bias = tf.Variable(
        tf.zeros(shape=[length]), name='b_' + str(BIAS_COUNTER))
    BIAS_COUNTER += 1
    return bias


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=2):
    global CONVOLUTION_COUNTER
    global POOLING_COUNTER
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME', 
                         name='conv_' + str(CONVOLUTION_COUNTER))
    CONVOLUTION_COUNTER += 1

    layer = tf.add(layer, biases)

    if pooling is not None and pooling > 1:
        layer = tf.nn.max_pool(value=layer, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME', 
                               name='pool_' + str(POOLING_COUNTER))
    POOLING_COUNTER += 1

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
