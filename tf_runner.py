import tensorflow as tf
import sys


IMAGE_WIDTH = 72
IMAGE_HEIGHT = 170
IMAGE_CHANNEL = 3

BATCH_SIZE = 100
LEARNING_RATE = 0.003

RANDOM_SEED = 1


def main():

    test = False

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

    if test is not True:
        layer_fc_1 = tf.nn.dropout(layer_fc_1, 0.5)

    layer_fc_2 = new_fc_layer(
        input=layer_fc_1, num_inputs=4096, num_outputs=4096)

    layer_fc_2 = tf.nn.sigmoid(layer_fc_2)

    if test is not True:
        layer_fc_2 = tf.nn.dropout(layer_fc_2, 0.5)

    layer_output = new_fc_layer(
        input=layer_fc_2, num_inputs=4096, num_outputs=1)

    layer_output = tf.nn.sigmoid(layer_output)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_output,
                                                            labels=output_placeholder)

    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

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

    if pooling > 1:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

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
    return layer

if __name__ == '__main__':
    main()
