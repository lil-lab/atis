"""Contains various utility functions for Dynet models."""

import dynet as dy
import numpy as np


def add_dim(variable, dim=0):
    """ Adds a dimension to a vector dy.Expression.

    Inputs:
        variable (dy.Expression): A vector Dynet expression.
        dim (int, optional): The dimension to add.

    Returns:
        dy.Expression with one more dimension (of size 1).
    """
    var_size = variable.dim()[0][0]

    if dim == 0:
        return dy.reshape(variable, (1, var_size))
    else:
        return dy.reshape(variable, (var_size, 1))


def is_vector(exp):
    """ Returns whether the expression is a vector.

    Inputs:
        exp (dy.Expression): The expression to check.

    Returns:
        bool, representing whether the expression is a vector.

    """
    return len(exp.dim()[0]) == 1

def split_exp(exp, num=2):
    """ Splits an expression into n parts.

    Inputs:
        exp (dy.Expression): A vector Dynet expression.
        num (int, optional): The number of parts to split it into.


    Returns:
        list of dy.Expression, containing the split expression.
    """
    assert is_vector(exp)
    size = exp.dim()[0][0]

    split_amount = int(size / num)

    parts = []
    for i in range(num):
        parts.append(exp[i * split_amount: (i + 1) * split_amount])

    return parts


def linear_transform(exp, params):
    """ Multiplies a dy.Expression and a set of parameters.

    Inputs:
        exp (dy.Expression): A Dynet tensor.
        params (dy.Parameters): Dynet parameters.

    Returns:
        dy.Expression representing exp * params.
    """
    if is_vector(exp):
        exp = add_dim(exp)

    return dy.transpose(exp) * params


def linear_layer(exp, weights, biases=None):
    """ Linear layer with weights and biases.

    Inputs:
        exp (dy.Expression): A Dynet tensor.
        params (dy.Parameters): Dynet parameters representing weights (a matrix).
        biases (dy.Parameters, optional): Dynet parameters representing biases
            (a vector).

    Returns:
        dy.Expression representing exp * weights + biases
    """
    if biases:
        return dy.affine_transform([add_dim(biases),
                                    add_dim(exp) if is_vector(exp) else exp,
                                    weights])
    else:
        return linear_transform(exp, weights)


def compute_loss(gold_seq,
                 scores,
                 index_to_token_maps,
                 gold_tok_to_id,
                 noise=0.00000001):
    """ Computes the loss of a gold sequence given scores.

    Inputs:
        gold_seq (list of str): A sequence of gold tokens.
        scores (list of dy.Expression): Expressions representing the scores of
            potential output tokens for each token in gold_seq.
        index_to_tok_maps (list of dict str->list of int): Maps from index in the
            sequence to a dictionary mapping from a string to a set of integers.
        gold_tok_to_id (lambda (str, str)->list of int): Maps from the gold token
            and some lookup function to the indices in the probability distribution
            where the gold token occurs.
        noise (float, optional): The amount of noise to add to the loss.

    Returns:
        dy.Expression representing the sum of losses over the sequence.
    """
    assert len(gold_seq) == len(scores)
    assert len(index_to_token_maps) == len(scores)

    losses = []
    for i, gold_tok in enumerate(gold_seq):
        score = scores[i]
        token_map = index_to_token_maps[i]

        gold_indices = gold_tok_to_id(gold_tok, token_map)
        assert len(gold_indices) > 0
        if len(gold_indices) == 1:
            losses.append(dy.pickneglogsoftmax(score, gold_indices[0]))
        else:
            prob_of_tok = dy.zeroes(1)
            probdist = dy.softmax(score)
            for index in gold_indices:
                prob_of_tok += probdist[index]
            prob_of_tok += noise
            losses.append(-dy.log(prob_of_tok))

    return dy.esum(losses)


def get_seq_from_scores(scores, index_to_token_maps):
    """Gets the argmax sequence from a set of scores.

    Inputs:
        scores (list of dy.Expression): Sequences of output scores.
        index_to_token_maps (list of list of str): For each output token, maps
            the index in the probability distribution to a string.

    Returns:
        list of str, representing the argmax sequence.
    """
    seq = []
    for score, tok_map in zip(scores, index_to_token_maps):
        assert score.dim()[0][0] == len(tok_map)
        seq.append(tok_map[np.argmax(score.npvalue())])
    return seq


def per_token_accuracy(gold_seq, pred_seq):
    """ Returns the per-token accuracy comparing two strings (recall).

    Inputs:
        gold_seq (list of str): A list of gold tokens.
        pred_seq (list of str): A list of predicted tokens.

    Returns:
        float, representing the accuracy.
    """
    num_correct = 0
    for i, gold_token in enumerate(gold_seq):
        if i < len(pred_seq) and pred_seq[i] == gold_token:
            num_correct += 1

    return float(num_correct) / len(gold_seq)

def get_utterances(item, history_size=1):
    """ Gets all of the relevant utterances for an example.

    Input:
        item (Utterance): The example.
        history_size (int, optional): The number of utterances to include.

    Returns:
        list of list of str, representing all of the sequences.
    """
    utterances = item.histories(history_size - 1)
    utterances.append(item.input_sequence())
    return utterances


def forward_one_multilayer(lstm_input, layer_states, dropout_amount=0.):
    """ Goes forward for one multilayer RNN cell step.

    Inputs:
        lstm_input (dy.Expression): Some input to the step.
        layer_states (list of dy.RNNState): The states of each layer in the cell.
        dropout_amount (float, optional): The amount of dropout to apply, in
            between the layers.

    Returns:
        (list of dy.Expression, list of dy.Expression), dy.Expression, (list of dy.RNNSTate),
        representing (each layer's cell memory, each layer's cell hidden state),
        the final hidden state, and (each layer's updated RNNState).
    """
    num_layers = len(layer_states)
    new_states = []
    cell_states = []
    hidden_states = []
    state = lstm_input
    for i in range(num_layers):
        new_states.append(layer_states[i].add_input(state))

        layer_c, layer_h = new_states[i].s()

        state = layer_h

        if i < num_layers - 1:
            state = dy.dropout(state, dropout_amount)

        cell_states.append(layer_c)
        hidden_states.append(layer_h)

    return (cell_states, hidden_states), state, new_states


def encode_sequence(sequence, rnns, embedder, dropout_amount=0.):
    """ Encodes a sequence given RNN cells and an embedding function.

    Inputs:
        seq (list of str): The sequence to encode.
        rnns (list of dy._RNNBuilder): The RNNs to use.
        emb_fn (dict str->dy.Expression): Function that embeds strings to
            word vectors.
        size (int): The size of the RNN.
        dropout_amount (float, optional): The amount of dropout to apply.

    Returns:
        (list of dy.Expression, list of dy.Expression), list of dy.Expression,
        where the first pair is the (final cell memories, final cell states) of
        all layers, and the second list is a list of the final layer's cell
        state for all tokens in the sequence.
    """
    layer_states = []
    for rnn in rnns:
        hidden_size = rnn.spec[2]
        layer_states.append(rnn.initial_state([dy.zeroes((hidden_size, 1)),
                                               dy.zeroes((hidden_size, 1))]))

    outputs = []

    for token in sequence:
        rnn_input = embedder(token)

        (cell_states, hidden_states), output, layer_states = \
            forward_one_multilayer(rnn_input,
                                   layer_states,
                                   dropout_amount)

        outputs.append(output)

    return (cell_states, hidden_states), outputs



def create_multilayer_lstm_params(num_layers,
                                  in_size,
                                  state_size,
                                  model,
                                  name=""):
    """ Adds a multilayer LSTM to the model parameters.

    Inputs:
        num_layers (int): Number of layers to create.
        in_size (int): The input size to the first layer.
        state_size (int): The size of the states.
        model (dy.ParameterCollection): The parameter collection for the model.
        name (str, optional): The name of the multilayer LSTM.
    """
    params = []
    in_size = in_size
    state_size = state_size
    for i in range(num_layers):
        layer_name = name + "-" + str(i)
        print(
            "LSTM " +
            layer_name +
            ": " +
            str(in_size) +
            " x " +
            str(state_size) +
            "; default Dynet initialization of hidden weights")
        params.append(dy.VanillaLSTMBuilder(1,
                                            in_size,
                                            state_size,
                                            model))
        in_size = state_size

    return params


def add_params(model, size, name=""):
    """ Adds parameters to the model.

    Inputs:
        model (dy.ParameterCollection): The parameter collection for the model.
        size (tuple of int): The size to create.
        name (str, optional): The name of the parameters.
    """
    if len(size) == 1:
        print("vector " + name + ": " +
              str(size[0]) + "; uniform in [-0.1, 0.1]")
    else:
        print("matrix " +
              name +
              ": " +
              str(size[0]) +
              " x " +
              str(size[1]) +
              "; uniform in [-0.1, 0.1]")
    return model.add_parameters(size,
                                init=dy.UniformInitializer(0.1),
                                name=name)
