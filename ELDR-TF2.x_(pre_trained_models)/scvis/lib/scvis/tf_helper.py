import tensorflow as tf

X_INIT = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")


def xavier(fan_in_out, name='weight'):
    return tf.Variable(X_INIT(fan_in_out), trainable=True, name=name)


def weight_xavier_relu(fan_in_out, name='weight'):
    stddev = tf.cast(tf.sqrt(2.0 / fan_in_out[0]), tf.float32)
    initial_w = tf.random.truncated_normal(shape=fan_in_out,
                                    mean=0.0, stddev=stddev)

    return tf.Variable(initial_w, trainable=True, name=name)


def weight_variable(fan_in_out, name='weight'):
    initial = tf.random.truncated_normal(shape=fan_in_out,
                                  mean=0.0, stddev=0.1)
    return tf.Variable(initial, trainable=True, name=name)


def bias_variable(fan_in_out, mean=0.1, name='bias'):
    initial = tf.constant(mean, shape=fan_in_out)
    return tf.Variable(initial, trainable=True, name=name)


def shape(tensor):
    return tensor.get_shape().as_list()
