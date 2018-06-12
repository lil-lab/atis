""" Class for the Sequence to sequence model for ATIS."""

import dynet as dy
import dynet_utils as du

from vocabulary import DEL_TOK, UNK_TOK

from decoder import SequencePredictor
from encoder import Encoder
from embedder import Embedder
from token_predictor import construct_token_predictor


def get_token_indices(token, index_to_token):
    """ Maps from a gold token (string) to a list of indices.

    Inputs:
        token (string): String to look up.
        index_to_token (list of tokens): Ordered list of tokens.

    Returns:
        list of int, representing the indices of the token in the probability
            distribution.
    """
    if token in index_to_token:
        if len(set(index_to_token)) == len(index_to_token):  # no duplicates
            return [index_to_token.index(token)]
        else:
            indices = []
            for index, other_token in enumerate(index_to_token):
                if token == other_token:
                    indices.append(index)
            assert len(indices) == len(set(indices))
            return indices
    else:
        return [index_to_token.index(UNK_TOK)]


def flatten_utterances(utterances):
    """ Gets a flat sequence from a sequence of utterances.

    Inputs:
        utterances (list of list of str): Utterances to concatenate.

    Returns:
        list of str, representing the flattened sequence with separating
            delimiter tokens.
    """
    sequence = []
    for i, utterance in enumerate(utterances):
        sequence.extend(utterance)
        if i < len(utterances) - 1:
            sequence.append(DEL_TOK)

    return sequence


def encode_snippets_with_states(snippets, states):
    """ Encodes snippets by using previous query states instead.

    Inputs:
        snippets (list of Snippet): Input snippets.
        states (list of dy.Expression): Previous hidden states to use.
        TODO: should this by dy.Expression or vector values?
    """
    for snippet in snippets:
        snippet.set_embedding(dy.concatenate([states[snippet.startpos],
                                              states[snippet.endpos]]))
    return snippets


class ATISModel():
    """ Sequence-to-sequence model for predicting a SQL query given an utterance
        and an interaction prefix.
    """

    def __init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            anonymizer):
        self.params = params

        self._pc = dy.ParameterCollection()

        # Create the input embeddings
        self.input_embedder = Embedder(self._pc,
                                       params.input_embedding_size,
                                       name="input-embedding",
                                       vocabulary=input_vocabulary,
                                       anonymizer=anonymizer)

        # Create the output embeddings
        self.output_embedder = Embedder(self._pc,
                                        params.output_embedding_size,
                                        name="output-embedding",
                                        vocabulary=output_vocabulary,
                                        anonymizer=anonymizer)

        # Create the encoder
        encoder_input_size = params.input_embedding_size
        if params.discourse_level_lstm:
            encoder_input_size += params.encoder_state_size / 2

        self.utterance_encoder = Encoder(params.encoder_num_layers,
                                         encoder_input_size,
                                         params.encoder_state_size,
                                         self._pc)

        # Positional embedder for utterances
        attention_key_size = params.encoder_state_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                self._pc,
                params.positional_embedding_size,
                name="positional-embedding",
                num_tokens=params.maximum_utterances)

        # Create the discourse-level LSTM parameters
        if params.discourse_level_lstm:
            self.discourse_lstms = du.create_multilayer_lstm_params(
                1, params.encoder_state_size, params.encoder_state_size / 2, self._pc, "LSTM-t")
            self.initial_discourse_state = du.add_params(self._pc, tuple(
                [params.encoder_state_size / 2]), "V-turn-state-0")

        # Snippet encoder
        final_snippet_size = 0
        if params.use_snippets and not params.previous_decoder_snippet_encoding:
            snippet_encoding_size = int(params.encoder_state_size / 2)
            final_snippet_size = params.encoder_state_size
            if params.snippet_age_embedding:
                snippet_encoding_size -= int(
                    params.snippet_age_embedding_size / 4)
                self.snippet_age_embedder = Embedder(
                    self._pc,
                    params.snippet_age_embedding_size,
                    name="snippet-age-embedding",
                    num_tokens=params.max_snippet_age_embedding)
                final_snippet_size = params.encoder_state_size \
                    + params.snippet_age_embedding_size / 2

            self.snippet_encoder = Encoder(params.snippet_num_layers,
                                           params.output_embedding_size,
                                           snippet_encoding_size,
                                           self._pc)
        token_predictor = construct_token_predictor(self._pc,
                                                    params,
                                                    output_vocabulary,
                                                    attention_key_size,
                                                    final_snippet_size,
                                                    anonymizer)

        self.decoder = SequencePredictor(
            params,
            params.output_embedding_size +
            attention_key_size,
            self.output_embedder,
            self._pc,
            token_predictor)

        self.trainer = dy.AdamTrainer(
            self._pc, alpha=params.initial_learning_rate)
        self.dropout = 0.

    def _encode_snippets(self,
                         previous_query,
                         snippets):
        """ Computes a single vector representation for each snippet.

        Inputs:
            previous_query (list of str): Previous query in the interaction.
            snippets (list of Snippet): Snippets extracted from the previous

        Returns:
            list of Snippets, where the embedding is set to a vector.
        """
        startpoints = [snippet.startpos for snippet in snippets]
        endpoints = [snippet.endpos for snippet in snippets]
        assert len(startpoints) == 0 or min(startpoints) >= 0
        assert len(endpoints) == 0 or max(endpoints) < len(previous_query)

        if previous_query and snippets:
            _, previous_outputs = self.snippet_encoder(
                previous_query, self.output_embedder, dropout_amount=self.dropout)
            assert len(previous_outputs) == len(previous_query)

            for snippet in snippets:
                embedding = dy.concatenate([previous_outputs[snippet.startpos],
                                            previous_outputs[snippet.endpos]])
                if self.params.snippet_age_embedding:
                    embedding = dy.concatenate([embedding, self.snippet_age_embedder(
                        min(snippet.age, self.params.max_snippet_age_embedding - 1))])
                snippet.set_embedding(embedding)

        return snippets

    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state

        discourse_lstm_states = [lstm.initial_state([dy.zeros((lstm.spec[2],)),
                                                     dy.zeros((lstm.spec[2],))])
                                 for lstm in self.discourse_lstms]
        return discourse_state, discourse_lstm_states

    def _encode_with_discourse_lstm(self, utterances):
        """ Encodes the utterances using a discourse-level LSTM, instead of concatenating.

        Inputs:
            utterances (list of list of str): Utterances.
        """
        hidden_states = []

        discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        final_state = None
        for utterance in utterances:
            final_state, utterance_states = self.utterance_encoder(
                utterance,
                lambda token: dy.concatenate([self.input_embedder(token), discourse_state]),
                dropout_amount=self.dropout)

            hidden_states.extend(utterance_states)

            _, discourse_state, discourse_lstm_states = du.forward_one_multilayer(
                final_state, discourse_lstm_states, self.dropout)

        return final_state, hidden_states

    def _encode_input_sequences(self, utterances):
        """ Encodes the input sequences.

        Inputs:
            utterances (list of list of str): Utterances to process.
        """
        if self.params.discourse_level_lstm:
            return self._encode_with_discourse_lstm(utterances)
        else:
            flat_utterances = flatten_utterances(utterances)
            final_state, hidden_states = self.utterance_encoder(
                flat_utterances, self.input_embedder, dropout_amount=self.dropout)

            states_no_delimiters = []
            start_utterance_index = 0
            for utterance in utterances:
                states_no_delimiters.extend(
                    hidden_states[start_utterance_index:start_utterance_index + len(utterance)])
                start_utterance_index += len(utterance) + 1

            return final_state, states_no_delimiters

    def _add_positional_embeddings(self, hidden_states, utterances):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(
                hidden_states[start_index:start_index + len(utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum([len(seq) for seq in grouped_states])

        assert sum([len(seq) for seq in grouped_states]) \
            == sum([len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []

        num_utterances_to_keep = min(
            self.params.maximum_utterances, len(utterances))
        for i, (states, utterance) in enumerate(zip(
                grouped_states[-num_utterances_to_keep:], utterances[-num_utterances_to_keep:])):
            positional_sequence = []
            index = num_utterances_to_keep - i - 1
            for state in states:
                positional_sequence.append(dy.concatenate(
                    [state, self.positional_embedder(index)]))

            assert len(positional_sequence) == len(utterance), \
                "Expected utterance and state sequence length to be the same, " \
                + "but they were " + str(len(utterance)) \
                + " and " + str(len(positional_sequence))
            new_states.extend(positional_sequence)
            flat_sequence.extend(utterance)
        return new_states, flat_sequence

    def train_step(self, batch):
        """Training step for a batch of examples.

        Input:
            batch (list of examples): Batch of examples used to update.
        """
        dy.renew_cg(autobatching=True)

        losses = []
        total_gold_tokens = 0

        batch.start()
        while not batch.done():
            example = batch.next()

            # First, encode the input sequences.
            input_sequences = example.histories(
                self.params.maximum_utterances - 1) + [example.input_sequence()]
            final_state, utterance_hidden_states = self._encode_input_sequences(
                input_sequences)

            # Add positional embeddings if appropriate
            if self.params.state_positional_embeddings:
                utterance_hidden_states = self._add_positional_embeddings(
                    utterance_hidden_states, input_sequences)

            # Encode the snippets
            snippets = []
            if self.params.use_snippets:
                snippets = self._encode_snippets(example.previous_query(), snippets)

            # Decode
            flat_seq = []
            for sequence in input_sequences:
                flat_seq.extend(sequence)
            decoder_results = self.decoder(
                final_state,
                utterance_hidden_states,
                self.params.train_maximum_sql_length,
                snippets=snippets,
                gold_sequence=example.gold_query(),
                dropout_amount=self.dropout,
                input_sequence=flat_seq)
            all_scores = [
                step.scores for step in decoder_results.predictions]
            all_alignments = [
                step.aligned_tokens for step in decoder_results.predictions]
            loss = du.compute_loss(example.gold_query(),
                                   all_scores,
                                   all_alignments,
                                   get_token_indices)
            losses.append(loss)
            total_gold_tokens += len(example.gold_query())

        average_loss = dy.esum(losses) / total_gold_tokens
        average_loss.forward()
        average_loss.backward()
        self.trainer.update()
        loss_scalar = average_loss.value()

        return loss_scalar

    def eval_step(self, example, feed_gold_query=False):
        """Evaluates the model on a specific example.

        Inputs:
            example (utterance example): Example to feed.
            feed_gold_query (bool): Whether or not to token-feed the gold query.
        """
        dy.renew_cg()
        # First, encode the input sequences.
        input_sequences = example.histories(
            self.params.maximum_utterances - 1) + [example.input_sequence()]
        final_state, utterance_hidden_states = self._encode_input_sequences(
            input_sequences)

        # Add positional embeddings if appropriate
        if self.params.state_positional_embeddings:
            utterance_hidden_states = self._add_positional_embeddings(
                utterance_hidden_states, input_sequences)

        # Encode the snippets
        snippets = []
        if self.params.use_snippets:
            snippets = self._encode_snippets(example.previous_query(), snippets)

        # Decode
        flat_seq = []
        for sequence in input_sequences:
            flat_seq.extend(sequence)
        decoder_results = self.decoder(
            final_state,
            utterance_hidden_states,
            self.params.train_maximum_sql_length,
            snippets=snippets,
            gold_sequence=example.gold_query() if feed_gold_query else None,
            dropout_amount=self.dropout,
            input_sequence=flat_seq)

        all_scores = [
            step.scores for step in decoder_results.predictions]
        all_alignments = [
            step.aligned_tokens for step in decoder_results.predictions]
        loss = dy.zeros(())
        if feed_gold_query:
            loss = du.compute_loss(example.gold_query(),
                                   all_scores,
                                   all_alignments,
                                   get_token_indices)
        predicted_seq = du.get_seq_from_scores(all_scores, all_alignments)
        return decoder_results, loss, predicted_seq

    def set_dropout(self, value):
        """ Sets the dropout to a specified value.

        Inputs:
            value (float): Value to set dropout to.
        """
        self.dropout = value

    def set_learning_rate(self, value):
        """ Sets the learning rate for the trainer.

        Inputs:
            value (float): The new learning rate.
        """
        self.trainer.learning_rate = value

    def save(self, filename):
        """ Saves the model to the specified filename.

        Inputs:
            filename (str): The filename to save to.
        """
        self._pc.save(filename)

    def load(self, filename):
        """ Loads saved parameters into the parameter collection.

        Inputs:
            filename (str): Name of file containing parameters.
        """
        self._pc.populate(filename)
        print("Loaded model from file " + filename)
