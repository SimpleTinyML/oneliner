from pathlib import Path

import numpy as np
import tensorflow as tf


def parameter(shape, index):
    values = (np.arange(np.prod(shape), dtype=np.float32) + index * 7) % 23
    return tf.constant(((values - 11) * 0.005).reshape(shape))


class LeNet5(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 32, 32, 1], tf.float32)])
    def main(self, value):
        parameters = [
            parameter(shape, index)
            for index, shape in enumerate(
                (
                    [5, 5, 1, 6],
                    [6],
                    [5, 5, 6, 16],
                    [16],
                    [400, 120],
                    [120],
                    [120, 84],
                    [84],
                    [84, 10],
                    [10],
                )
            )
        ]
        value = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(value, parameters[0], 1, "VALID"), parameters[1]
            )
        )
        value = tf.nn.max_pool2d(value, 2, 2, "VALID")
        value = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(value, parameters[2], 1, "VALID"), parameters[3]
            )
        )
        value = tf.nn.max_pool2d(value, 2, 2, "VALID")
        value = tf.reshape(value, [1, 400])
        value = tf.nn.relu(tf.matmul(value, parameters[4]) + parameters[5])
        value = tf.nn.relu(tf.matmul(value, parameters[6]) + parameters[7])
        return tf.matmul(value, parameters[8]) + parameters[9]


def main():
    model = LeNet5()
    output = Path(__file__).with_name("lenet5_tensorflow")
    tf.saved_model.save(
        model,
        output,
        signatures={"serving_default": model.main.get_concrete_function()},
    )
    print(model.main(tf.ones([1, 32, 32, 1])).numpy().reshape(-1).tolist())


if __name__ == "__main__":
    main()
