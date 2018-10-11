"""Contains a main function for training and/or evaluating a model."""

import os

import numpy as np

from pycrayon import CrayonClient
from slackclient import SlackClient
from slackclient.exceptions import SlackClientError

from parse_args import interpret_args

import atis_data
from interaction_model import InteractionATISModel
from logger import Logger
from model import ATISModel
from model_util import Metrics, evaluate_utterance_sample, evaluate_interaction_sample, \
    train_epoch_with_utterances, train_epoch_with_interactions, evaluate_using_predicted_queries
from visualize_attention import AttentionGraph


VALID_EVAL_METRICS = [
    Metrics.LOSS,
    Metrics.TOKEN_ACCURACY,
    Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY]
FINAL_EVAL_METRICS = [
    Metrics.STRING_ACCURACY,
    Metrics.TOKEN_ACCURACY,
    Metrics.CORRECT_TABLES,
    Metrics.STRICT_CORRECT_TABLES,
    Metrics.SYNTACTIC_QUERIES,
    Metrics.SEMANTIC_QUERIES]


def send_slack_message(username, message, channel):
    """Sends a message to your Slack channel.

    Input:
        username (str): Username to send from.
        message (str): The message to send.
        channel (str): Channel to send the message to.
    """
    token = ''  # TODO: put your Slack token here.
    try:
        client = SlackClient(token)
        client.api_call(
            'chat.postMessage',
            channel=channel,
            text=message,
            username=username,
            icon_emoji=':robot_face:')
    except SlackClientError as error:
        print("Couldn't send slack message with exception " + str(error))


def train(model, data, params):
    """ Trains a model.

    Inputs:
        model (ATISModel): The model to train.
        data (ATISData): The data that is used to train.
        params (namespace): Training parameters.
    """
    # Get the training batches.
    log = Logger(os.path.join(params.logdir, params.logfile), "w")
    num_train_original = atis_data.num_utterances(data.train_data)
    log.put("Original number of training utterances:\t"
            + str(num_train_original))

    eval_fn = evaluate_utterance_sample
    trainbatch_fn = data.get_utterance_batches
    trainsample_fn = data.get_random_utterances
    validsample_fn = data.get_all_utterances
    batch_size = params.batch_size
    if params.interaction_level:
        batch_size = 1
        eval_fn = evaluate_interaction_sample
        trainbatch_fn = data.get_interaction_batches
        trainsample_fn = data.get_random_interactions
        validsample_fn = data.get_all_interactions

    maximum_output_length = params.train_maximum_sql_length
    train_batches = trainbatch_fn(batch_size,
                                  max_output_length=maximum_output_length,
                                  randomize=not params.deterministic)

    if params.num_train >= 0:
        train_batches = train_batches[:params.num_train]

    training_sample = trainsample_fn(params.train_evaluation_size,
                                     max_output_length=maximum_output_length)
    valid_examples = validsample_fn(data.valid_data,
                                    max_output_length=maximum_output_length)

    num_train_examples = sum([len(batch) for batch in train_batches])
    num_steps_per_epoch = len(train_batches)

    log.put(
        "Actual number of used training examples:\t" +
        str(num_train_examples))
    log.put("(Shortened by output limit of " +
            str(maximum_output_length) +
            ")")
    log.put("Number of steps per epoch:\t" + str(num_steps_per_epoch))
    log.put("Batch size:\t" + str(batch_size))

    print(
        "Kept " +
        str(num_train_examples) +
        "/" +
        str(num_train_original) +
        " examples")
    print(
        "Batch size of " +
        str(batch_size) +
        " gives " +
        str(num_steps_per_epoch) +
        " steps per epoch")

    # Keeping track of things during training.
    epochs = 0
    patience = params.initial_patience
    learning_rate_coefficient = 1.
    previous_epoch_loss = float('inf')
    maximum_validation_accuracy = 0.
    maximum_string_accuracy = 0.
    crayon = CrayonClient(hostname="localhost")
    experiment = crayon.create_experiment(params.logdir)

    countdown = int(patience)

    keep_training = True
    while keep_training:
        log.put("Epoch:\t" + str(epochs))
        model.set_dropout(params.dropout_amount)
        model.set_learning_rate(
            learning_rate_coefficient *
            params.initial_learning_rate)

        # Run a training step.
        if params.interaction_level:
            epoch_loss = train_epoch_with_interactions(
                train_batches,
                params,
                model,
                randomize=not params.deterministic)
        else:
            epoch_loss = train_epoch_with_utterances(
                train_batches,
                model,
                randomize=not params.deterministic)

        log.put("train epoch loss:\t" + str(epoch_loss))
        experiment.add_scalar_value("train_loss", epoch_loss, step=epochs)

        model.set_dropout(0.)

        # Run an evaluation step on a sample of the training data.
        train_eval_results = eval_fn(training_sample,
                                     model,
                                     params.train_maximum_sql_length,
                                     "train-eval",
                                     gold_forcing=True,
                                     metrics=TRAIN_EVAL_METRICS)[0]

        for name, value in train_eval_results.items():
            log.put(
                "train final gold-passing " +
                name.name +
                ":\t" +
                "%.2f" %
                value)
            experiment.add_scalar_value(
                "train_gold_" + name.name, value, step=epochs)

        # Run an evaluation step on the validation set.
        valid_eval_results = eval_fn(valid_examples,
                                     model,
                                     "valid-eval",
                                     gold_forcing=True,
                                     metrics=VALID_EVAL_METRICS)[0]
        for name, value in valid_eval_results.items():
            log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)
            experiment.add_scalar_value(
                "valid_gold_" + name.name, value, step=epochs)

        valid_loss = valid_eval_results[Metrics.LOSS]
        valid_token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
        string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

        if valid_loss > previous_epoch_loss:
            learning_rate_coefficient *= params.learning_rate_ratio
            log.put(
                "learning rate coefficient:\t" +
                str(learning_rate_coefficient))
        experiment.add_scalar_value(
            "learning_rate",
            learning_rate_coefficient,
            step=epochs)
        previous_epoch_loss = valid_loss
        saved = False
        if valid_token_accuracy > maximum_validation_accuracy:
            saved = True
            maximum_validation_accuracy = valid_token_accuracy
            patience = patience * params.patience_ratio
            countdown = int(patience)
            last_save_file = os.path.join(params.logdir, "save_" + str(epochs))
            model.save(last_save_file)

            log.put("maximum accuracy:\t" + str(maximum_validation_accuracy))
            log.put("patience:\t" + str(patience))
            log.put("save file:\t" + str(last_save_file))
        if not saved and string_accuracy > maximum_string_accuracy:
            maximum_string_accuracy = string_accuracy
            log.put(
                "maximum string accuracy:\t" +
                str(maximum_string_accuracy))
            last_save_file = os.path.join(params.logdir, "save_" + str(epochs))
            model.save(last_save_file)

        break
        send_slack_message(
            username=params.logdir,
            message="Epoch " +
            str(epochs) +
            ": " +
            str(string_accuracy) +
            " validation accuracy; countdown is " +
            str(countdown),
            channel="models")

        if countdown <= 0:
            keep_training = False

        countdown -= 1
        log.put("countdown:\t" + str(countdown))
        experiment.add_scalar_value("countdown", countdown, step=epochs)
        log.put("")

        epochs += 1

    log.put("Finished training!")
    send_slack_message(username=params.logdir,
                       message="Done training!!",
                       channel="@alsuhr")
    log.close()

    return last_save_file


def evaluate(model, data, params, last_save_file):
    """Evaluates a pretrained model on a dataset.

    Inputs:
        model (ATISModel): Model class.
        data (ATISData): All of the data.
        params (namespace): Parameters for the model.
        last_save_file (str): Location where the model save file is.
    """
    if last_save_file:
        model.load(last_save_file)
    else:
        if not params.save_file:
            raise ValueError(
                "Must provide a save file name if not training first.")
        model.load(params.save_file)

    split = None
    if params.evaluate_split == 'dev':
        split = data.dev_data
    elif params.evaluate_split == 'train':
        split = data.train_data
    elif params.evaluate_split == 'test':
        split = data.test_data
    elif params.evaluate_split == 'valid':
        split = data.valid_data
    else:
        raise ValueError("Split not recognized: " + str(params.evaluate_split))

    filename = params.evaluate_split
    if params.use_predicted_queries:
        filename += "predicted"
    else:
        filename += "gold"

    full_name = os.path.join(params.logdir, filename) + params.results_note

    if params.interaction_level or params.use_predicted_queries:
        examples = data.get_all_interactions(split)
        if params.interaction_level:
            evaluate_interaction_sample(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout,
                use_predicted_queries=params.use_predicted_queries,
                max_generation_length=params.eval_maximum_sql_length,
                write_results=True,
                use_gpu=True)
        else:
            evaluate_using_predicted_queries(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout)
    else:
        examples = data.get_all_utterances(split)
        evaluate_utterance_sample(
            examples,
            model,
            name=full_name,
            gold_forcing=False,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            max_generation_length=params.eval_maximum_sql_length,
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            write_results=True)


def evaluate_attention(model, data, params, last_save_file):
    """Evaluates attention distributions during generation.

    Inputs:
        model (ATISModel): The model.
        data (ATISData): Data to evaluate.
        params (namespace): Parameters for the run.
        last_save_file (str): The save file to load from.
    """
    if not params.save_file:
        raise ValueError(
            "Must provide a save file name for evaluating attention.")
    model.load(last_save_file)

    all_data = data.get_all_interactions(data.dev_data)
    found_one = None
    for interaction in all_data:
        if interaction.identifier.replace("/", "") == params.reference_results:
            found_one = interaction
            break

    data = [found_one]

    # Do analysis on the random example.
    ignore_with_gpu = [line.strip() for line in open(
        "cpu_full_interactions.txt").readlines()]
    for interaction in data:
        if interaction.identifier in ignore_with_gpu:
            continue
        identifier = interaction.identifier.replace("/", "")
        full_path = os.path.join(params.logdir, identifier)

        if os.path.exists(full_path):
            continue
        if params.use_predicted_queries:
            predictions = model.predict_with_predicted_queries(
                interaction, params.eval_maximum_sql_length, syntax_restrict=True)
        else:
            predictions = model.predict_with_gold_queries(
                interaction, params.eval_maximum_sql_length)

        for i, prediction in enumerate(predictions):
            item = interaction.gold_utterances()[i]
            input_sequence = [token for utterance in item.histories(
                params.maximum_utterances - 1) + [item.input_sequence()] for token in utterance]
            attention_graph = AttentionGraph(input_sequence)

            if params.use_predicted_queries:
                item = interaction.processed_utterances[i]

            output_sequence = prediction[0]
            attentions = [
                result.attention_results for result in prediction[-1].predictions]

            for token, attention in zip(output_sequence, attentions):
                attention_graph.add_attention(
                    token, np.transpose(
                        attention.distribution.value())[0])
            suffix = identifier + "_attention_" + str(i) + ".tex"
            filename = os.path.join(full_path, suffix)

            if not os.path.exists(full_path):
                os.mkdir(full_path)
            attention_graph.render_as_latex(filename)
            os.system(
                "cd " +
                str(full_path) +
                "; pdflatex " +
                suffix +
                "; rm *.log; rm *.aux")
            print("rendered " + str(filename))


def interact(model, params, anonymizer, last_save_file=""):
    """Interactive command line tool.

    Inputs:
        model (ATISModel): The model to interact with.
        params (namespace): Parameters for the run.
        anonymizer (Anonymizer): Class for anonymizing user input.
        last_save_file (str): The save file to load from.
    """
    if last_save_file:
        model.load(last_save_file)
    else:
        if not params.save_file:
            raise ValueError(
                "Must provide a save file name if not training first.")
        model.load(params.save_file)

    model.interactive_prediction(anonymizer)


def main():
    """Main function that trains and/or evaluates a model."""
    params = interpret_args()

    # Prepare the dataset into the proper form.
    data = atis_data.ATISDataset(params)

    # Construct the model object.
    model_type = InteractionATISModel if params.interaction_level else ATISModel

    model = model_type(
        params,
        data.input_vocabulary,
        data.output_vocabulary,
        data.anonymizer if params.anonymize and params.anonymization_scoring else None)

    last_save_file = ""

    if params.train:
        last_save_file = train(model, data, params)
    if params.evaluate:
        evaluate(model, data, params, last_save_file)
    if params.interactive:
        interact(model, params, data.anonymizer, last_save_file)
    if params.attention:
        evaluate_attention(model, data, params, params.save_file)


if __name__ == "__main__":
    main()
