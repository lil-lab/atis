""" Class for the Sequence to sequence model for ATIS."""

import dynet as dy
import dynet_utils as du
import vocabulary as vocab

def gold_tok_to_id(token, idx_to_token):
    """ Maps from a gold token to a list of indices in the probability distribution.

    Inputs:
        token (int): The unique ID of the token.
        idx_to_token (dict int->str): Maps from indices in the probability
            distribution to strings.
    """
    if token in idx_to_token:
        if len(set(idx_to_token)) == len(
                idx_to_token):  # no duplicates
            return [idx_to_token.index(token)]
        else:
            indices = []
            for index, check_tok in enumerate(idx_to_token):
                if token == check_tok:
                    indices.append(index)
            assert len(indices) == len(set(indices))
            return indices
    else:
        return [idx_to_token.index(vocab.UNK_TOK)]

def predict(model,
            utterances,
            prev_query=None,
            snippets=None,
            gold_seq=None,
            dropout_amount=0.,
            loss_only=False,
            beam_size=1.):
    """ Predicts a SQL query given an utterance and other various inputs.

    Inputs:
        model (Seq2SeqModel): The model to use to predict.
        utterances (list of list of str): The utterances to predict for.
        prev_query (list of str, optional): The previously generated query.
        snippets (list of Snippet. optional): The snippets available for prediction.
        all_snippets (list of Snippet, optional): All snippets so far in the interaction.
        gold_seq (list of str, optional): The gold sequence.
        dropout_amount (float, optional): How much dropout to apply during predictino.
        loss_only (bool, optional): Whether to only return the loss.
        beam_size (float, optional): How many items to include in the beam during prediction.
    """
    assert len(prev_query) == 0 or model.use_snippets
    assert len(snippets) == 0 or model.use_snippets
    assert not loss_only or len(gold_seq) > 0

    (enc_state, enc_outputs), input_seq = model.encode_input_sequences(
        utterances, dropout_amount)

    embedded_snippets = []
    if snippets:
        embedded_snippets = model.encode_snippets(
            prev_query, snippets, dropout_amount=dropout_amount)
        assert len(embedded_snippets) == len(snippets)

    if gold_seq:
        item = model.decode(
            enc_state,
            enc_outputs,
            input_seq,
            snippets=embedded_snippets if model.use_snippets else [],
            gold_seq=gold_seq,
            dropout_amount=dropout_amount)[0]
        scores = item.scores
        scores_by_timestep = [score[0] for score in scores]
        score_maps_by_timestep = [score[1] for score in scores]

        assert scores_by_timestep[0].dim()[0][0] == len(
            score_maps_by_timestep[0])
        assert len(score_maps_by_timestep[0]) >= len(model.output_vocab) + len(snippets)

        loss = du.compute_loss(gold_seq,
                               scores_by_timestep,
                               score_maps_by_timestep,
                               gold_tok_to_id,
                               noise=0.00000000001)

        if loss_only:
            return loss
        sequence = du.get_seq_from_scores(scores_by_timestep,
                                          score_maps_by_timestep)
    else:
        item = model.decode(
            enc_state,
            enc_outputs,
            input_seq,
            snippets=embedded_snippets if model.use_snippets else [],
            beam_size=beam_size)[0]
        scalar_loss = 0
        sequence = item.sequence

    token_acc = 0
    if gold_seq:
        token_acc = du.per_token_accuracy(gold_seq, sequence)

    return sequence, scalar_loss, token_acc, item.probability


def prepare_and_predict(model,
                        item,
                        use_gold=False,
                        training=False,
                        dropout=0.,
                        beam_size=1):
    utterances = du.get_utterances(
        item, model.input_vocab, model.history_length)
    assert len(utterances) <= model.history_length
    if use_gold:
        assert item.flatten_sequence(
            item.gold_query()) == item.original_gold_query()
    return model.predict(
        utterances,
        prev_query=item.previous_query() if model.use_snippets else [],
        snippets=item.snippets() if model.use_snippets else [],
        gold_seq=item.gold_query() if use_gold else [],
        loss_only=training,
        dropout_amount=dropout,
        beam_size=1 if training else beam_size)


def train_step(model, batch, lr_coeff, dropout):
    dy.renew_cg()
    losses = []
    assert not model.prev_decoder_snippet_rep

    num_tokens = 0
    for item in batch.items:
        loss = model.prepare_and_predict(item,
                                         use_gold=True,
                                         training=True,
                                         dropout=dropout)
        num_tokens += len(item.gold_query())
        losses.append(loss)

    final_loss = dy.esum(losses) / num_tokens
    final_loss.forward()
    final_loss.backward()
    model.trainer.learning_rate = lr_coeff
    model.trainer.update()

    return final_loss.npvalue()[0]


# eval_step
# Runs an evaluation on the example.
#
# Inputs:
#    example: an Utterance.
#    use_gold: whether or not to pass gold tokens into the decoder.
#
# Outputs:
#    information provided by prepare and predict
def eval_step(model,
              example,
              use_gold=False,
              dropout_amount=0.,
              beam_size=1):
    dy.renew_cg()
    assert not model.prev_decoder_snippet_rep
    return model.prepare_and_predict(example,
                                     use_gold=use_gold,
                                     training=False,
                                     dropout=dropout_amount,
                                     beam_size=beam_size)
