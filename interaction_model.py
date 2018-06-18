""" Class for the Sequence to sequence model for ATIS."""

import dynet as dy
import dynet_utils as du
import snippets as snippet_handler
import sql_util
import vocabulary as vocab
import tokenizers

from model import ATISModel, encode_snippets_with_states, get_token_indices
from utterance import ANON_INPUT_KEY


LIMITED_INTERACTIONS = {"raw/atis2/12-1.1/ATIS2/TEXT/TRAIN/SRI/QS0/1": 22,
                        "raw/atis3/17-1.1/ATIS3/SP_TRN/MIT/8K7/5": 14,
                        "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5": -1}

END_OF_INTERACTION = {"quit", "exit", "done"}


class InteractionATISModel(ATISModel):
    """ Interaction ATIS model, where an interaction is processed all at once.
    """

    def __init__(self,
                 params,
                 input_vocabulary,
                 output_vocabulary,
                 anonymizer):
        ATISModel.__init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            anonymizer)

    def predict_turn(self,
                     utterance_final_state,
                     input_hidden_states,
                     max_generation_length,
                     gold_query=None,
                     snippets=None,
                     input_sequence=None,
                     feed_gold_tokens=False,
                     training=False):
        """ Gets a prediction for a single turn -- calls decoder and updates loss, etc.

        TODO:  this can probably be split into two methods, one that just predicts
            and another that computes the loss.
        """
        predicted_sequence = []
        fed_sequence = []
        loss = None
        token_accuracy = 0.

        if feed_gold_tokens:
            decoder_results = self.decoder(utterance_final_state,
                                           input_hidden_states,
                                           max_generation_length,
                                           gold_sequence=gold_query,
                                           input_sequence=input_sequence,
                                           snippets=snippets,
                                           dropout_amount=self.dropout)

            all_scores = [
                step.scores for step in decoder_results.predictions]
            all_alignments = [
                step.aligned_tokens for step in decoder_results.predictions]
            # Compute the loss
            loss = du.compute_loss(gold_query,
                                   all_scores,
                                   all_alignments,
                                   get_token_indices)

            if not training:
                predicted_sequence = du.get_seq_from_scores(
                    all_scores, all_alignments)

                token_accuracy = du.per_token_accuracy(
                    gold_query, predicted_sequence)

            fed_sequence = gold_query
        else:
            decoder_results = self.decoder(utterance_final_state,
                                           input_hidden_states,
                                           max_generation_length,
                                           input_sequence=input_sequence,
                                           snippets=snippets,
                                           dropout_amount=self.dropout)
            predicted_sequence = decoder_results.sequence

            fed_sequence = predicted_sequence

        # fed_sequence contains EOS, which we don't need when encoding snippets.
        # also ignore the first state, as it contains the BEG encoding.
        decoder_states = [
            pred.decoder_state for pred in decoder_results.predictions]

        for token, state in zip(fed_sequence[:-1], decoder_states[1:]):
            if snippet_handler.is_snippet(token):
                snippet_length = 0
                for snippet in snippets:
                    if snippet.name == token:
                        snippet_length = len(snippet.sequence)
                        break
                assert snippet_length > 0
                decoder_states.extend([state for _ in range(snippet_length)])
            else:
                decoder_states.append(state)

        return (predicted_sequence,
                loss,
                token_accuracy,
                decoder_states,
                decoder_results)

    def train(self,
              interaction,
              max_generation_length,
              snippet_alignment_probability=1.):
        """ Trains the interaction-level model on a single interaction.

        Inputs:
            interaction (Interaction): The interaction to train on.
            learning_rate (float): Learning rate to use.
            snippet_keep_age (int): Age of oldest snippets to use.
            snippet_alignment_probability (float): The probability that a snippet will
                be used in constructing the gold sequence.
        """
        assert self.params.discourse_level_lstm

        dy.renew_cg()

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []
        final_utterance_state = None

        decoder_states = []

        discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        for utterance_index, utterance in enumerate(
                interaction.gold_utterances()):
            if interaction.identifier in LIMITED_INTERACTIONS \
                    and utterance_index > LIMITED_INTERACTIONS[interaction.identifier]:
                break

            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Get the gold query: reconstruct if the alignment probability
            # is less than one
            if snippet_alignment_probability < 1.:
                gold_query = sql_util.add_snippets_to_query(
                    available_snippets,
                    utterance.contained_entities(),
                    utterance.anonymized_gold_query(),
                    prob_align=snippet_alignment_probability) + [vocab.EOS_TOK]
            else:
                gold_query = utterance.gold_query()

            # Encode the utterance, and update the discourse-level states
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                lambda token: dy.concatenate([self.input_embedder(token), discourse_state]),
                dropout_amount=self.dropout)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            _, discourse_state, discourse_lstm_states = du.forward_one_multilayer(
                final_utterance_state[1][0], discourse_lstm_states, self.dropout)

            flat_sequence = []
            num_utterances_to_keep = min(
                self.params.maximum_utterances,
                len(input_sequences))
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(
                        available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(
                        previous_query, available_snippets)

            if len(gold_query) <= max_generation_length \
                    and len(previous_query) <= max_generation_length:
                prediction = self.predict_turn(final_utterance_state,
                                               utterance_states,
                                               max_generation_length,
                                               gold_query=gold_query,
                                               snippets=snippets,
                                               input_sequence=flat_sequence,
                                               feed_gold_tokens=True,
                                               training=True)
                loss = prediction[1]
                decoder_states = prediction[3]
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # Break if previous decoder snippet encoding -- because the previous
                # sequence was too long to run the decoder.
                if self.params.previous_decoder_snippet_encoding:
                    break
                continue

        if losses:
            average_loss = dy.esum(losses) / total_gold_tokens

            # Renormalize so the effect is normalized by the batch size.
            normalized_loss = average_loss
            if self.params.reweight_batch:
                normalized_loss = len(losses) * average_loss / \
                    float(self.params.batch_size)
            normalized_loss.forward()
            normalized_loss.backward()
            self.trainer.update()
            loss_scalar = normalized_loss.value()
        else:
            loss_scalar = 0.

        return loss_scalar

    def predict_with_predicted_queries(
            self,
            interaction,
            max_generation_length,
            syntax_restrict=True):
        """ Predicts an interaction, using the predicted queries to get snippets."""
        assert self.params.discourse_level_lstm

        dy.renew_cg()

        predictions = []

        input_hidden_states = []
        input_sequences = []
        final_utterance_state = None

        discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        interaction.start_interaction()
        while not interaction.done():
            utterance = interaction.next_utterance()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            input_sequence = utterance.input_sequence()
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                lambda token: dy.concatenate([self.input_embedder(token), discourse_state]))

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            _, discourse_state, discourse_lstm_states = du.forward_one_multilayer(
                final_utterance_state[1][0], discourse_lstm_states)

            flat_sequence = []
            num_utterances_to_keep = min(
                self.params.maximum_utterances,
                len(input_sequences))
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)

            snippets = None
            if self.params.use_snippets:
                snippets = self._encode_snippets(previous_query, available_snippets)

            results = self.predict_turn(final_utterance_state,
                                        utterance_states,
                                        max_generation_length,
                                        input_sequence=flat_sequence,
                                        snippets=snippets)

            predicted_sequence = results[0]
            predictions.append(results)

            # Update things necessary for using predicted queries
            anonymized_sequence = utterance.remove_snippets(predicted_sequence)[
                :-1]
            flat_sequence = utterance.flatten_sequence(predicted_sequence)

            if not syntax_restrict or sql_util.executable(
                    flat_sequence,
                    username=self.params.database_username,
                    password=self.params.database_password,
                    timeout=self.params.database_timeout):
                utterance.set_pred_query(
                    interaction.remove_snippets(predicted_sequence))
                interaction.add_utterance(
                    utterance,
                    anonymized_sequence,
                    previous_snippets=utterance.snippets())

            else:
                utterance.set_predicted_query(utterance.previous_query())
                interaction.add_utterance(
                    utterance,
                    utterance.previous_query(),
                    previous_snippets=utterance.snippets())

        return predictions

    def predict_with_gold_queries(self,
                                  interaction,
                                  max_generation_length,
                                  feed_gold_query=False):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        assert self.params.discourse_level_lstm

        dy.renew_cg()

        predictions = []

        input_hidden_states = []
        input_sequences = []
        final_utterance_state = None

        decoder_states = []

        discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        for utterance in interaction.gold_utterances():
            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Encode the utterance, and update the discourse-level states
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                lambda token: dy.concatenate([self.input_embedder(token), discourse_state]),
                dropout_amount=self.dropout)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            _, discourse_state, discourse_lstm_states = du.forward_one_multilayer(
                final_utterance_state[1][0], discourse_lstm_states, self.dropout)

            flat_sequence = []
            num_utterances_to_keep = min(
                self.params.maximum_utterances,
                len(input_sequences))
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(
                        available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(
                        previous_query, available_snippets)

            prediction = self.predict_turn(final_utterance_state,
                                           utterance_states,
                                           max_generation_length,
                                           gold_query=utterance.gold_query(),
                                           snippets=snippets,
                                           input_sequence=flat_sequence,
                                           feed_gold_tokens=feed_gold_query)
            decoder_states = prediction[3]
            predictions.append(prediction)

        return predictions

    def interactive_prediction(self, anonymizer):
        """Interactive prediction.

        Inputs:
            anonymizer (Anonymizer): Anonymizer to use for user's input.
        """
        dy.renew_cg()

        snippet_bank = []
        anonymization_dictionary = {}
        previous_query = []

        input_hidden_states = []
        input_sequences = []
        final_utterance_state = None

        discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        utterance = "show me flights from new york to boston"  # input("> ")
        while utterance.lower() not in END_OF_INTERACTION:

            # First, need to normalize the utterance and get an anonymization
            # dictionary.
            tokenized_sequence = tokenizers.nl_tokenize(utterance)

            available_snippets = [
                snippet for snippet in snippet_bank if snippet.index <= 1]

            sequence_to_use = tokenized_sequence

            # TODO: implement date normalization

            if self.params.anonymize:
                sequence_to_use = anonymizer.anonymize(
                    tokenized_sequence,
                    anonymization_dictionary,
                    ANON_INPUT_KEY,
                    add_new_anon_toks=True)

            # Now we encode the sequence
            final_utterance_state, utterance_states = self.utterance_encoder(
                sequence_to_use,
                lambda token: dy.concatenate([self.input_embedder(token), discourse_state]))

            input_hidden_states.extend(utterance_states)
            input_sequences.append(sequence_to_use)

            # Now update the discourse state
            _, discourse_state, discourse_lstm_states = du.forward_one_multilayer(
                final_utterance_state[1][0], discourse_lstm_states)

            # Add positional embeddings
            flat_sequence = []
            num_utterances_to_keep = min(
                self.params.maximum_utterances,
                len(input_sequences))
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)

            # Encode the snippets
            if self.params.use_snippets:
                snippets = self._encode_snippets(previous_query, available_snippets)

            # Predict a result
            results = self.predict_turn(final_utterance_state,
                                        utterance_states,
                                        self.params.eval_maximum_sql_length,
                                        input_sequence=flat_sequence,
                                        snippets=snippets)

            # Get the sequence, and show the de-anonymized and flattened
            # versions
            predicted_sequence = results[0]
            print(predicted_sequence)

            # Execute the query and show the results

            # Update the available snippets, etc.

        utterance = input("> ")
