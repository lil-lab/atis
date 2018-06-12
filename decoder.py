""" Decoder for the SQL generation problem."""

from collections import namedtuple
import numpy as np

import dynet as dy
import dynet_utils as du

from token_predictor import PredictionInput

from vocabulary import EOS_TOK, UNK_TOK

def flatten_distribution(distribution_map, probabilities):
    """ Flattens a probability distribution given a map of "unique" values.
        All values in distribution_map with the same value should get the sum
        of the probabilities.

        Arguments:
            distribution_map (list of str): List of values to get the probability for.
            probabilities (np.ndarray): Probabilities corresponding to the values in
                distribution_map.

        Returns:
            list, np.ndarray of the same size where probabilities for duplicates
                in distribution_map are given the sum of the probabilities in probabilities.
    """
    assert len(distribution_map) == len(probabilities)
    if len(distribution_map) != len(set(distribution_map)):
        idx_first_dup = 0
        seen_set = set()
        for i, tok in enumerate(distribution_map):
            if tok in seen_set:
                idx_first_dup = i
                break
            seen_set.add(tok)
        new_dist_map = distribution_map[:idx_first_dup] + list(
            set(distribution_map) - set(distribution_map[:idx_first_dup]))
        assert len(new_dist_map) == len(set(new_dist_map))
        new_probs = np.array(
            probabilities[:idx_first_dup] \
            + [0. for _ in range(len(set(distribution_map)) \
                                 - idx_first_dup)])
        assert len(new_probs) == len(new_dist_map)

        for i, token_name in enumerate(
                distribution_map[idx_first_dup:]):
            if token_name not in new_dist_map:
                new_dist_map.append(token_name)

            new_index = new_dist_map.index(token_name)
            new_probs[new_index] += probabilities[i +
                                                  idx_first_dup]
        new_probs = new_probs.tolist()
    else:
        new_dist_map = distribution_map
        new_probs = probabilities

    assert len(new_dist_map) == len(new_probs)

    return new_dist_map, new_probs

class SQLPrediction(namedtuple('SQLPrediction',
                               ('predictions',
                                'sequence',
                                'probability'))):
    """Contains prediction for a sequence."""
    __slots__ = ()

    def __str__(self):
        return str(self.probability) + "\t" + " ".join(self.sequence)

class SequencePredictor():
    """ Predicts a sequence.

    Attributes:
        lstms (list of dy.RNNBuilder): The RNN used.
        token_predictor (TokenPredictor): Used to actually predict tokens.
    """
    def __init__(self,
                 params,
                 input_size,
                 output_embedder,
                 model,
                 token_predictor):
        self.lstms = du.create_multilayer_lstm_params(
            params.decoder_num_layers, input_size, params.decoder_state_size, model, "LSTM-d")
        self.token_predictor = token_predictor
        self.output_embedder = output_embedder
        self.start_token_embedding = du.add_params(model,
                                                   (params.output_embedding_size,),
                                                   "y-0")

    def _initialize_decoder_lstm(self, encoder_state):
        decoder_lstm_states = []
        for i, lstm in enumerate(self.lstms):
            encoder_layer_num = 0
            if len(encoder_state[0]) > 1:
                encoder_layer_num = i
            decoder_lstm_states.append(
                lstm.initial_state(
                    (encoder_state[0][encoder_layer_num],
                     encoder_state[1][encoder_layer_num])))
        return decoder_lstm_states

    def __call__(self,
                 final_encoder_state,
                 encoder_states,
                 max_generation_length,
                 snippets=None,
                 gold_sequence=None,
                 input_sequence=None,
                 dropout_amount=0.):
        """ Generates a sequence. """
        index = 0

        context_vector_size = self.token_predictor.attention_module.value_size

        # Decoder states: just the initialized decoder.
        # Current input to decoder: phi(start_token) ; zeros the size of the
        # context vector
        predictions = []
        sequence = []
        probability = 1.

        decoder_states = self._initialize_decoder_lstm(final_encoder_state)
        decoder_input = dy.concatenate([self.start_token_embedding,
                                        dy.zeroes((context_vector_size,))])

        continue_generating = True

        while continue_generating:
            if len(sequence) == 0 or sequence[-1] != EOS_TOK:
                _, decoder_state, decoder_states = du.forward_one_multilayer(
                    decoder_input, decoder_states, dropout_amount)
                prediction_input = PredictionInput(decoder_state=decoder_state,
                                                   input_hidden_states=encoder_states,
                                                   snippets=snippets,
                                                   input_sequence=input_sequence)
                prediction = self.token_predictor(prediction_input,
                                                  dropout_amount=dropout_amount)

                predictions.append(prediction)

                if gold_sequence:
                    decoder_input = dy.concatenate(
                        [self.output_embedder.bow_snippets(gold_sequence[index],
                                                           snippets),
                         prediction.attention_results.vector])
                    sequence.append(gold_sequence[index])

                    if index >= len(gold_sequence) - 1:
                        continue_generating = False
                else:
                    probabilities = np.transpose(dy.softmax(
                        prediction.scores).npvalue()).tolist()[0]
                    distribution_map = prediction.aligned_tokens

                    # Get a new probabilities and distribution_map consolidating
                    # duplicates
                    distribution_map, probabilities = flatten_distribution(distribution_map,
                                                                           probabilities)

                    # Modify the probability distribution so that the UNK token can
                    # never be produced
                    probabilities[distribution_map.index(UNK_TOK)] = 0.
                    argmax_index = int(np.argmax(probabilities))

                    argmax_token = distribution_map[argmax_index]
                    sequence.append(argmax_token)

                    decoder_input = dy.concatenate(
                        [self.output_embedder.bow_snippets(argmax_token, snippets),
                         prediction.attention_results.vector])
                    probability *= probabilities[argmax_index]

                    continue_generating = False
                    if index < max_generation_length and argmax_token != EOS_TOK:
                        continue_generating = True

            index += 1

        return SQLPrediction(predictions,
                             sequence,
                             probability)
