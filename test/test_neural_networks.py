from finalproject import neural_networks as nn
import os
import numpy as np


def test_autoencoder():
    binary_input = np.asarray([list(map(lambda x: float(x),
                                        list(num))) for num in [
                                            bin(x)[2:].zfill(n) for x in [
                                                int(np.power(2, i)
                                                    ) for i in range(0, 8)]]])
    layer_sizes = [8, 3, 8]
    reg_rate = 0.0
    num_epochs = 4000
    num_batches = 2
    learn_rate = 1.0
    weights, biases, sse = nn.train_network(num_epochs, binary_input.T,
                                            binary_input.T, learning_rate,
                                            reg_rate, layer_sizes, num_batches)
    output = np.round(feed_forward(weights_list[0], biases_list[0],
                                   binary_input.T).T, 1)
    assert np.array_equal(binary_input, output)


def test_dna_to_binary():
    # tractable subset
    DNA_TO_BIN = {'a':'111', 't':'100', 'c':'010', 'g':'001'}
    seq = 'atcg'
    output = nn.dna_to_binary(seq)
    assert np.array_equal(output, np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]))
