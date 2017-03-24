import numpy as np
import random


# sigmoid function
# can evaluate numpy arrays
# COMPLETE
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


# feed-forward network
# evaluate the network output for a given input
# COMPLETE
def feed_forward(weights, biases, inp):
    input_var = inp
    for w, b in zip(weights, biases):
        inp = sigmoid(np.dot(w, inp) + np.tile(b, (1, inp.shape[1])))
    return inp


# update the gradient descent based on the current minibatch
# compute the change in weights and biases based on a single element of the minibatch
# sum all of the changes to get the final change in weight and bias for the whole minibatch
# update the weights and biases
# COMPLETE
def eval_whole_batch(weights, biases, batch_inp, batch_outp, rate, reg_param):
    del_grad_weights, del_grad_biases = backpropagate(weights, biases, batch_inp, batch_outp)
    del_grad_biases = [np.array([np.sum(x, axis=1)]).T for x in del_grad_biases]
    weights = np.array([weight - rate * (
        change / batch_inp.shape[1] + reg_param * weight) for weight, change in zip(weights, del_grad_weights)])
    biases = np.array([bias - (rate / batch_inp.shape[1]) * change for bias, change in zip(biases, del_grad_biases)])
    return weights, biases


# backpropagation
# compute the values for adjusting the weights and biases
# to minimize the error (diff between expected and actual ouput)
# COMPLETE
def backpropagate(weights, biases, inp, outp):
    # feed forward
    input_var = inp
    activation_mat = [input_var]
    out_mat = []
    for w, b in zip(weights, biases):
        out_mat.append(np.dot(w, input_var) + np.tile(b, (1, input_var.shape[1])))
        activation_mat.append(sigmoid(out_mat[-1]))
        input_var = activation_mat[-1]
    # back-propagate
    delta_mat = [0.0 for _ in range(len(weights))]
    del_grad_weights = [np.zeros(weight.shape) for weight in weights]
    del_grad_biases = [np.zeros(bias.shape) for bias in biases]
    delta_mat[-1] = l1_dist(activation_mat[-1], outp)
    del_grad_weights[-1] = np.dot(delta_mat[-1], activation_mat[-2].T)
    del_grad_biases[-1] = delta_mat[-1]
    for i in range(2, len(weights) + 1):
        delta_mat[-i] = np.dot(weights[(-i) + 1].T, delta_mat[(-i) + 1]) * (
            activation_mat[-i] * (1.0 - activation_mat[-i]))
        del_grad_weights[-i] = np.dot(delta_mat[-i], activation_mat[-(i + 1)].T)
        del_grad_biases[-i] = delta_mat[-i]
    return del_grad_weights, del_grad_biases


# calculate L1 distance
def l1_dist(a, b):
    return a - b


# calculate hamming distance
def hamming_dist(a, b):
    return np.count_nonzero(a ^ b)


# calculate the sum of squared errors for two data sets, samples in columns
def calc_sse(mat1, mat2):
    return np.sum(np.sqrt(np.square(mat1 - mat2))) / (mat1.shape[1])


# mini-batch gradient descent
# calculate the next step toward
# some minimum...hopefully a global one
# COMPLETE
def train_network(num_epochs, batch_inp, batch_outp, learning_rate, reg_rate, layer_sizes, batch_size):
    epochs = range(num_epochs)
    weight_mats = [
        np.random.normal(0, 1.0 / np.sqrt(layer_sizes[i]), (layer_sizes[i + 1], layer_sizes[i])) for i in range(0, len(layer_sizes) - 1)]
    bias_mats = [np.random.normal(0, 0.0001, (layer_sizes[i], 1)) for i in range(1, len(layer_sizes))]
    sse = []
    for i in epochs:
        mini_batches = zip(np.array_split(batch_inp, num_batches, axis=1),
                           np.array_split(batch_outp, num_batches, axis=1))
        for split_in, split_out in mini_batches:
            weight_mats, bias_mats = eval_whole_batch(
                weight_mats, bias_mats, split_in, split_out, learning_rate, reg_rate)
        sse.append(calc_sse(batch_outp, feed_forward(weight_mats, bias_mats, batch_inp)))
    return weight_mats, bias_mats, sse

DNA_TO_BIN = {'a':'111', 't':'100', 'c':'010', 'g':'001'}
BIN_TO_DNA = {'111':'a', '100':'t', '010':'c', '001':'g'}


# convert dna sequence to binary sequence
def dna_to_binary(seq):
    return np.array(list(map(lambda x: int(x), ''.join([DNA_TO_BIN[str.lower(x)] for x in list(seq)]))))


# convert sequence lists to training/testing examples
def create_case(seq_list, is_positive_boolean):
    return [(x, is_positive_boolean) for x in seq_list]


# read in positive/test sets
# COMPLETE
def read_txt(fn):
    output = set([])
    with open(fn, 'r') as f:
        for line in f:
            output.add(line.strip())
    return output


# read in negative examples
# COMPLETE
def read_fasta(fn):
    output = set([])
    with open(fn, 'r') as f:
        next(f)
        current = ''
        print(len(current))
        for line in f:
            if line[0] == '>' and len(current) > 0:
                if '>' in current:
                    print(line)
                    print(current)
                output.add(str(current))
                current = ''
            else:
                current += line.strip()
    return output


# convert each negative sequence to length 17 sequences
def chop_negatives(seq_set):
    output_set = set([])
    while len(seq_set) > 0:
        el = seq_set.pop()
        output_set.update([el[i:i + 17] for i in range(len(el) - 17)])
    return output_set


# remove intersections between positive and negative sets,
# but keep instance in positive
# COMPLETE
def clean_negatives(pos_set, neg_set):
    neg_set.difference_update(pos_set)
    return neg_set


# k-fold validation
# given a set of samples, shuffle the order and split it into lists of size 1/k and k-1/k
def create_k_fold(inp_list, k):
    random.shuffle(inp_list)
    sep_point = int(len(inp_list) / k)
    print(sep_point)
    return inp_list[:sep_point], inp_list[sep_point:]


# write output to a file
def write_output(seq_list, score_list, fn):
    out_string = ''
    for seq, val in zip(list(seq_list), score_list):
        out_string += seq + '\t' + str(val) + '\t'
    out_string = out_string[:-2]
    with open(fn, 'w') as f:
        f.write(out_string)


# generate positive and negative training cases
def gen_cases(neg_fn, pos_fn):
    pos_seqs = list(read_txt(pos_fn))
    neg_seqs = read_fasta(neg_fn)
    neg_seqs = chop_negatives(neg_seqs)
    neg_seqs = list(clean_negatives(pos_seqs, neg_seqs))
    return np.array([dna_to_binary(x) for x in pos_seqs]), np.array([dna_to_binary(x) for x in neg_seqs])
