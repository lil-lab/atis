"""Predicts a token."""

from collections import namedtuple

import dynet as dy
import dynet_utils as du

from attention import Attention

class PredictionInput(namedtuple('PredictionInput',
                                 ('decoder_state',
                                  'input_hidden_states',
                                  'snippets',
                                  'input_sequence'))):
    """ Inputs to the token predictor. """
    __slots__ = ()


class TokenPrediction(namedtuple('TokenPrediction',
                                 ('scores',
                                  'aligned_tokens',
                                  'attention_results',
                                  'decoder_state'))):

    """A token prediction.

    Attributes:
        scores (dy.Expression): Scores for each possible output token.
        aligned_tokens (list of str): The output tokens, aligned with the scores.
        attention_results (AttentionResult): The result of attending on the input
            sequence.
    """
    __slots__ = ()


def score_snippets(snippets, scorer):
    """ Scores snippets given a scorer.

    Inputs:
        snippets (list of Snippet): The snippets to score.
        scorer (dy.Expression): Dynet vector against which to score  the snippets.

    Returns:
        dy.Expression, list of str, where the first is the scores and the second
            is the names of the snippets that were scored.
    """
    snippet_expressions = [snippet.embedding for snippet in snippets]
    all_snippet_embeddings = dy.concatenate(snippet_expressions, d=1)

    if du.is_vector(scorer):
        scorer = du.add_dim(scorer)

    scores = dy.transpose(dy.transpose(scorer) * all_snippet_embeddings)

    if scores.dim()[0][0] != len(snippets):
        raise ValueError("Got " + str(scores.dim()[0][0]) + " scores for "
                         + str(len(snippets)) + " snippets")

    return scores, [snippet.name for snippet in snippets]


class TokenPredictor():
    """ Predicts a token given a (decoder) state.

    Attributes:
        vocabulary (Vocabulary): A vocabulary object for the output.
        attention_module (Attention): An attention module.
        state_transformation_weights (dy.Parameters): Transforms the input state
            before predicting a token.
        vocabulary_weights (dy.Parameters): Final layer weights.
        vocabulary_biases (dy.Parameters): Final layer biases.
    """

    def __init__(self, model, params, vocabulary, attention_key_size):
        self.vocabulary = vocabulary
        self.attention_module = Attention(model,
                                          params.decoder_state_size,
                                          attention_key_size,
                                          attention_key_size)
        self.state_transform_weights = du.add_params(
            model,
            (params.decoder_state_size +
             attention_key_size,
             params.decoder_state_size),
            "weights-state-transform")
        self.vocabulary_weights = du.add_params(
            model, (params.decoder_state_size, len(vocabulary)), "weights-vocabulary")
        self.vocabulary_biases = du.add_params(model,
                                               tuple([len(vocabulary)]),
                                               "biases-vocabulary")

    def _get_intermediate_state(self, state, dropout_amount=0.):
        intermediate_state = dy.tanh(
            du.linear_layer(
                state, self.state_transform_weights))
        return dy.dropout(intermediate_state, dropout_amount)

    def _score_vocabulary_tokens(self, state):
        scores = dy.transpose(du.linear_layer(state,
                                              self.vocabulary_weights,
                                              self.vocabulary_biases))
        if scores.dim()[0][0] != len(self.vocabulary.inorder_tokens):
            raise ValueError("Got " +
                             str(scores.dim()[0][0]) +
                             " scores for " +
                             str(len(self.vocabulary.inorder_tokens)) +
                             " vocabulary items")

        return scores, self.vocabulary.inorder_tokens

    def __call__(self,
                 prediction_input,
                 dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = dy.concatenate(
            [decoder_state, attention_results.vector])

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        return TokenPrediction(vocab_scores, vocab_tokens, attention_results, decoder_state)


class SnippetTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts snippets.

    Attributes:
        snippet_weights (dy.Parameter): Weights for scoring snippets against some
            state.
    """

    def __init__(
            self,
            model,
            params,
            vocabulary,
            attention_key_size,
            snippet_size):
        TokenPredictor.__init__(self,
                                model,
                                params,
                                vocabulary,
                                attention_key_size)
        if snippet_size <= 0:
            raise ValueError("Snippet size must be greater than zero; was " \
                + str(snippet_size))
        self.snippet_weights = du.add_params(model,
                                             (params.decoder_state_size,
                                              snippet_size),
                                             "weights-snippet")

    def _get_snippet_scorer(self, state):
        return dy.transpose(du.linear_layer(dy.transpose(state),
                                            self.snippet_weights))

    def __call__(self,
                 prediction_input,
                 dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        snippets = prediction_input.snippets

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = dy.concatenate(
            [decoder_state, attention_results.vector])

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        if snippets:
            snippet_scores, snippet_tokens = score_snippets(
                snippets,
                self._get_snippet_scorer(intermediate_state))

            final_scores = dy.concatenate([final_scores, snippet_scores])
            aligned_tokens.extend(snippet_tokens)

        return TokenPrediction(final_scores,
                               aligned_tokens,
                               attention_results,
                               decoder_state)


class AnonymizationTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts anonymization tokens.

    Attributes:
        anonymizer (Anonymizer): The anonymization object.

    """

    def __init__(self,
                 model,
                 params,
                 vocabulary,
                 attention_key_size,
                 anonymizer):
        TokenPredictor.__init__(self,
                                model,
                                params,
                                vocabulary,
                                attention_key_size)
        if not anonymizer:
            raise ValueError("Expected an anonymizer, but was None")
        self.anonymizer = anonymizer

    def _score_anonymized_tokens(self,
                                 input_sequence,
                                 attention_scores):
        scores = []
        tokens = []
        for i, token in enumerate(input_sequence):
            if self.anonymizer.is_anon_tok(token):
                scores.append(attention_scores[i])
                tokens.append(token)

        if len(scores) > 0:
            if len(scores) != len(tokens):
                raise ValueError("Got " + str(len(scores)) + " scores for "
                                 + str(len(tokens)) + " anonymized tokens")

            return dy.concatenate(scores), tokens
        else:
            return None, []

    def __call__(self,
                 prediction_input,
                 dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        input_sequence = prediction_input.input_sequence
        assert input_sequence

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = dy.concatenate(
            [decoder_state, attention_results.vector])

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        anonymized_scores, anonymized_tokens = self._score_anonymized_tokens(
            input_sequence,
            attention_results.scores)

        if anonymized_scores:
            final_scores = dy.concatenate([final_scores, anonymized_scores])
            aligned_tokens.extend(anonymized_tokens)

        return TokenPrediction(final_scores,
                               aligned_tokens,
                               attention_results,
                               decoder_state)


class SnippetAnonymizationTokenPredictor(
        SnippetTokenPredictor,
        AnonymizationTokenPredictor):
    """ Token predictor that both anonymizes and scores snippets."""

    def __init__(self,
                 model,
                 params,
                 vocabulary,
                 attention_key_size,
                 snippet_size,
                 anonymizer):
        SnippetTokenPredictor.__init__(self,
                                       model,
                                       params,
                                       vocabulary,
                                       attention_key_size,
                                       snippet_size)
        AnonymizationTokenPredictor.__init__(self,
                                             model,
                                             params,
                                             vocabulary,
                                             attention_key_size,
                                             anonymizer)

    def __call__(self,
                 prediction_input,
                 dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        assert prediction_input.input_sequence

        snippets = prediction_input.snippets

        attention_results = self.attention_module(decoder_state,
                                                  prediction_input.input_hidden_states)

        intermediate_state = self._get_intermediate_state(
            dy.concatenate([decoder_state, attention_results.vector]),
            dropout_amount=dropout_amount)

        # Vocabulary tokens
        final_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        # Snippets
        if snippets:
            snippet_scores, snippet_tokens = score_snippets(
                snippets,
                self._get_snippet_scorer(intermediate_state))

            final_scores = dy.concatenate([final_scores, snippet_scores])
            aligned_tokens.extend(snippet_tokens)

        # Anonymized tokens
        anonymized_scores, anonymized_tokens = self._score_anonymized_tokens(
            prediction_input.input_sequence,
            attention_results.scores)

        if anonymized_scores:
            final_scores = dy.concatenate([final_scores, anonymized_scores])
            aligned_tokens.extend(anonymized_tokens)

        return TokenPrediction(final_scores,
                               aligned_tokens,
                               attention_results,
                               decoder_state)


def construct_token_predictor(parameter_collection,
                              params,
                              vocabulary,
                              attention_key_size,
                              snippet_size,
                              anonymizer=None):
    """ Constructs a token predictor given the parameters.

    Inputs:
        parameter_collection (dy.ParameterCollection): Contains the parameters.
        params (dictionary): Contains the command line parameters/hyperparameters.
        vocabulary (Vocabulary): Vocabulary object for output generation.
        attention_key_size (int): The size of the attention keys.
        anonymizer (Anonymizer): An anonymization object.
    """
    if params.use_snippets and anonymizer and not params.previous_decoder_snippet_encoding:
        return SnippetAnonymizationTokenPredictor(parameter_collection,
                                                  params,
                                                  vocabulary,
                                                  attention_key_size,
                                                  snippet_size,
                                                  anonymizer)
    elif params.use_snippets and not params.previous_decoder_snippet_encoding:
        return SnippetTokenPredictor(parameter_collection,
                                     params,
                                     vocabulary,
                                     attention_key_size,
                                     snippet_size)
    elif anonymizer:
        return AnonymizationTokenPredictor(parameter_collection,
                                           params,
                                           vocabulary,
                                           attention_key_size,
                                           anonymizer)
    else:
        return TokenPredictor(parameter_collection,
                              params,
                              vocabulary,
                              attention_key_size)
