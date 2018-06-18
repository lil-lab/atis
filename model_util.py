"""Basic model training and evaluation functions."""

from enum import Enum

import random
import sys

import progressbar

import dynet_utils as du
import metrics as metrics_handler
import sql_util


class Metrics(Enum):
    """Definitions of simple metrics to compute."""
    LOSS = 1
    TOKEN_ACCURACY = 2
    STRING_ACCURACY = 3
    CORRECT_TABLES = 4
    STRICT_CORRECT_TABLES = 5
    SEMANTIC_QUERIES = 6
    SYNTACTIC_QUERIES = 7


def get_progressbar(name, size):
    """Gets a progress bar object given a name and the total size.

    Inputs:
        name (str): The name to display on the side.
        size (int): The maximum size of the progress bar.

    """
    return progressbar.ProgressBar(maxval=size,
                                   widgets=[name,
                                            progressbar.Bar('=', '[', ']'),
                                            ' ',
                                            progressbar.Percentage(),
                                            ' ',
                                            progressbar.ETA()])


def train_epoch_with_utterances(batches,
                                model,
                                randomize=True):
    """Trains model for a single epoch given batches of utterance data.

    Inputs:
        batches (UtteranceBatch): The batches to give to training.
        model (ATISModel): The model obect.
        learning_rate (float): The learning rate to use during training.
        dropout_amount (float): Amount of dropout to set in the model.
        randomize (bool): Whether or not to randomize the order that the batches are seen.
    """
    if randomize:
        random.shuffle(batches)
    progbar = get_progressbar("train     ", len(batches))
    progbar.start()
    loss_sum = 0.

    for i, batch in enumerate(batches):
        batch_loss = model.train_step(batch)
        loss_sum += batch_loss

        progbar.update(i)

    progbar.finish()

    total_loss = loss_sum / len(batches)

    return total_loss


def train_epoch_with_interactions(interaction_batches,
                                  params,
                                  model,
                                  randomize=True):
    """Trains model for single epoch given batches of interactions.

    Inputs:
        interaction_batches (list of InteractionBatch): The batches to train on.
        params (namespace): Parameters to run with.
        model (ATISModel): Model to train.
        randomize (bool): Whether or not to randomize the order that batches are seen.
    """
    if randomize:
        random.shuffle(interaction_batches)
    progbar = get_progressbar("train     ", len(interaction_batches))
    progbar.start()
    loss_sum = 0.

    for i, interaction_batch in enumerate(interaction_batches):
        assert len(interaction_batch) == 1
        interaction = interaction_batch.items[0]

        if interaction.identifier == "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5":
            continue
        try:
            batch_loss = model.train(
                interaction, params.train_maximum_sql_length)
        except RuntimeError as exception:
            print("Failed on interaction: " + str(interaction.identifier))
            print(exception)
            print("\n\n")
        loss_sum += batch_loss

        progbar.update(i)

    progbar.finish()

    total_loss = loss_sum / len(interaction_batches)

    return total_loss


def update_sums(metrics,
                metrics_sums,
                predicted_sequence,
                flat_sequence,
                gold_query,
                original_gold_query,
                gold_forcing=False,
                loss=None,
                token_accuracy=0.,
                database_username="",
                database_password="",
                database_timeout=0,
                gold_table=None):
    """" Updates summing for metrics in an aggregator.

    TODO: don't use sums, just keep the raw value.
    """
    if Metrics.LOSS in metrics:
        metrics_sums[Metrics.LOSS] += loss.npvalue()[0]
    if Metrics.TOKEN_ACCURACY in metrics:
        if gold_forcing:
            metrics_sums[Metrics.TOKEN_ACCURACY] += token_accuracy
        else:
            num_tokens_correct = 0.
            for j, token in enumerate(gold_query):
                if len(
                        predicted_sequence) > j and predicted_sequence[j] == token:
                    num_tokens_correct += 1
            metrics_sums[Metrics.TOKEN_ACCURACY] += num_tokens_correct / \
                len(gold_query)
    if Metrics.STRING_ACCURACY in metrics:
        metrics_sums[Metrics.STRING_ACCURACY] += int(
            flat_sequence == original_gold_query)

    if Metrics.CORRECT_TABLES in metrics:
        assert database_username, "You did not provide a database username"
        assert database_password, "You did not provide a database password"
        assert database_timeout > 0, "Database timeout is 0 seconds"

        # Evaluate SQL
        if flat_sequence != original_gold_query:
            syntactic, semantic, table = sql_util.execution_results(
                " ".join(flat_sequence), database_username, database_password, database_timeout)
        else:
            syntactic = True
            semantic = True
            table = gold_table

        metrics_sums[Metrics.CORRECT_TABLES] += int(table == gold_table)
        if Metrics.SYNTACTIC_QUERIES in metrics:
            metrics_sums[Metrics.SYNTACTIC_QUERIES] += int(syntactic)
        if Metrics.SEMANTIC_QUERIES in metrics:
            metrics_sums[Metrics.SEMANTIC_QUERIES] += int(semantic)
        if Metrics.STRICT_CORRECT_TABLES in metrics:
            metrics_sums[Metrics.STRICT_CORRECT_TABLES] += int(
                table == gold_table and syntactic)


def construct_averages(metrics_sums, total_num):
    """ Computes the averages for metrics.

    Inputs:
        metrics_sums (dict Metric -> float): Sums for a metric.
        total_num (int): Number to divide by (average).
    """
    metrics_averages = {}
    for metric, value in metrics_sums.items():
        metrics_averages[metric] = value / total_num
        if metric != "loss":
            metrics_averages[metric] *= 100.

    return metrics_averages


def evaluate_utterance_sample(sample,
                              model,
                              max_generation_length,
                              name="",
                              gold_forcing=False,
                              metrics=None,
                              total_num=-1,
                              database_username="",
                              database_password="",
                              database_timeout=0,
                              write_results=False):
    """Evaluates a sample of utterance examples.

    Inputs:
        sample (list of Utterance): Examples to evaluate.
        model (ATISModel): Model to predict with.
        max_generation_length (int): Maximum length to generate.
        name (str): Name to log with.
        gold_forcing (bool): Whether to force the gold tokens during decoding.
        metrics (list of Metric): Metrics to evaluate with.
        total_num (int): Number to divide by when reporting results.
        database_username (str): Username to use for executing queries.
        database_password (str): Password to use when executing queries.
        database_timeout (float): Timeout on queries when executing.
        write_results (bool): Whether to write the results to a file.
    """
    assert metrics

    if total_num < 0:
        total_num = len(sample)

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with filename " + str(name) + "_predictions.json")
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    predictions = []
    for i, item in enumerate(sample):
        _, loss, predicted_seq = model.eval_step(
            item, max_generation_length, feed_gold_query=gold_forcing)
        loss = loss / len(item.gold_query())
        predictions.append(predicted_seq)

        flat_sequence = item.flatten_sequence(predicted_seq)
        token_accuracy = du.per_token_accuracy(
            item.gold_query(), predicted_seq)

        if write_results:
            metrics_handler.write_prediction(
                predictions_file,
                identifier=item.interaction.identifier,
                input_seq=item.input_sequence(),
                probability=0,
                prediction=predicted_seq,
                flat_prediction=flat_sequence,
                gold_query=item.gold_query(),
                flat_gold_queries=item.original_gold_queries(),
                gold_tables=item.gold_tables(),
                index_in_interaction=item.utterance_index,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout)

        update_sums(metrics,
                    metrics_sums,
                    predicted_seq,
                    flat_sequence,
                    item.gold_query(),
                    item.original_gold_queries()[0],
                    gold_forcing,
                    loss,
                    token_accuracy,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=item.gold_tables()[0])

        progbar.update(i)

    progbar.finish()
    predictions_file.close()

    return construct_averages(metrics_sums, total_num), None


def evaluate_interaction_sample(sample,
                                model,
                                max_generation_length,
                                name="",
                                gold_forcing=False,
                                metrics=None,
                                total_num=-1,
                                database_username="",
                                database_password="",
                                database_timeout=0,
                                use_predicted_queries=False,
                                write_results=False,
                                use_gpu=False):
    """ Evaluates a sample of interactions. """
    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with file " + str(name + "_predictions.json"))
    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    num_utterances = 0
    ignore_with_gpu = [line.strip() for line in open(
        "cpu_full_interactions.txt").readlines()]
    predictions = []

    use_gpu = not ("--no_gpus" in sys.argv or "--no_gpus=1" in sys.argv)
    for i, interaction in enumerate(sample):
        if use_gpu and interaction.identifier in ignore_with_gpu:
            continue
        elif not use_gpu and interaction.identifier not in ignore_with_gpu:
            continue
        try:
            if use_predicted_queries:
                example_preds = model.predict_with_predicted_queries(
                    interaction,
                    max_generation_length)
            else:
                example_preds = model.predict_with_gold_queries(
                    interaction,
                    max_generation_length,
                    feed_gold_query=gold_forcing)
        except RuntimeError as exception:
            print("Failed on interaction: " + str(interaction.identifier))
            print(exception)
            print("\n\n")
            exit()

        predictions.extend(example_preds)

        assert len(example_preds) == len(
            interaction.interaction.utterances) or not example_preds
        for j, pred in enumerate(example_preds):
            num_utterances += 1

            sequence, loss, token_accuracy, _, decoder_results = pred

            if use_predicted_queries:
                item = interaction.processed_utterances[j]
                original_utt = interaction.interaction.utterances[item.index]
                gold_query = original_utt.gold_query_to_use
                original_gold_query = original_utt.original_gold_query
                gold_table = original_utt.gold_sql_results
                gold_queries = [q[0] for q in original_utt.all_gold_queries]
                gold_tables = [q[1] for q in original_utt.all_gold_queries]
                index = item.index
            else:
                item = interaction.gold_utterances()[j]
                gold_query = item.gold_query()
                original_gold_query = item.original_gold_query()
                gold_table = item.gold_table()
                gold_queries = item.original_gold_queries()
                gold_tables = item.gold_tables()
                index = item.utterance_index
            if loss:
                loss = loss / len(gold_query)

            flat_sequence = item.flatten_sequence(sequence)

            if write_results:
                metrics_handler.write_prediction(
                    predictions_file,
                    identifier=interaction.identifier,
                    input_seq=item.input_sequence(),
                    probability=decoder_results.probability,
                    prediction=sequence,
                    flat_prediction=flat_sequence,
                    gold_query=gold_query,
                    flat_gold_queries=gold_queries,
                    gold_tables=gold_tables,
                    index_in_interaction=index,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout)

            update_sums(metrics,
                        metrics_sums,
                        sequence,
                        flat_sequence,
                        gold_query,
                        original_gold_query,
                        gold_forcing,
                        loss,
                        token_accuracy,
                        database_username=database_username,
                        database_password=database_password,
                        database_timeout=database_timeout,
                        gold_table=gold_table)

        progbar.update(i)

    progbar.finish()

    if total_num < 0:
        total_num = num_utterances

    predictions_file.close()
    return construct_averages(metrics_sums, total_num), predictions


def evaluate_using_predicted_queries(sample,
                                     model,
                                     name="",
                                     gold_forcing=False,
                                     metrics=None,
                                     total_num=-1,
                                     database_username="",
                                     database_password="",
                                     database_timeout=0,
                                     snippet_keep_age=1):
    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with file " + str(name + "_predictions.json"))
    assert not gold_forcing
    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    num_utterances = 0
    predictions = []
    for i, item in enumerate(sample):
        int_predictions = []
        item.start_interaction()
        while not item.done():
            utterance = item.next_utterance(snippet_keep_age)

            predicted_sequence, loss, _, probability = model.eval_step(
                utterance)
            int_predictions.append((utterance, predicted_sequence))

            flat_sequence = utterance.flatten_sequence(predicted_sequence)

            if sql_util.executable(
                    flat_sequence,
                    username=database_username,
                    password=database_password,
                    timeout=database_timeout) and probability >= 0.24:
                utterance.set_pred_query(
                    item.remove_snippets(predicted_sequence))
                item.add_utterance(utterance,
                                   item.remove_snippets(predicted_sequence),
                                   previous_snippets=utterance.snippets())
            else:
                # Add the /previous/ predicted query, guaranteed to be syntactically
                # correct
                seq = []
                utterance.set_pred_query(seq)
                item.add_utterance(
                    utterance, seq, previous_snippets=utterance.snippets())

            original_utt = item.interaction.utterances[utterance.index]
            metrics_handler.write_prediction(
                predictions_file,
                identifier=item.interaction.identifier,
                input_seq=utterance.input_sequence(),
                probability=probability,
                prediction=predicted_sequence,
                flat_prediction=flat_sequence,
                gold_query=original_utt.gold_query_to_use,
                flat_gold_queries=[
                    q[0] for q in original_utt.all_gold_queries],
                gold_tables=[
                    q[1] for q in original_utt.all_gold_queries],
                index_in_interaction=utterance.index,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout)

            update_sums(metrics,
                        metrics_sums,
                        predicted_sequence,
                        flat_sequence,
                        original_utt.gold_query_to_use,
                        original_utt.original_gold_query,
                        gold_forcing,
                        loss,
                        token_accuracy=0,
                        database_username=database_username,
                        database_password=database_password,
                        database_timeout=database_timeout,
                        gold_table=original_utt.gold_sql_results)

        predictions.append(int_predictions)
        progbar.update(i)

    progbar.finish()

    if total_num < 0:
        total_num = num_utterances
    predictions_file.close()

    return construct_averages(metrics_sums, total_num), predictions
