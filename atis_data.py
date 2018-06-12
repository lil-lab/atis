""" Utility functions for loading and processing ATIS data."""
import anonymization as anon
import atis_batch
import copy
import dataset_split as ds
import os
import random

from interaction import load_function
from entities import NLtoSQLDict
from atis_vocab import ATISVocabulary

ENTITIES_FILENAME = 'entities.txt'
ANONYMIZATION_FILENAME = 'anonymization.txt'

class ATISDataset():
    """ Contains the ATIS data. """
    def __init__(self, params):
        self.anonymizer = None
        if params.anonymize:
            self.anonymizer = anon.Anonymizer(ANONYMIZATION_FILENAME)

        if not os.path.exists(params.data_directory):
            os.mkdir(params.data_directory)

        self.entities_dictionary = NLtoSQLDict(ENTITIES_FILENAME)

        int_load_function = load_function(params,
                                          self.entities_dictionary,
                                          self.anonymizer)

        self.train_data = ds.DatasetSplit(
            os.path.join(params.data_directory, params.processed_train_filename),
            params.raw_train_filename,
            int_load_function)
        if params.train:
            self.valid_data = ds.DatasetSplit(
                os.path.join(params.data_directory, params.processed_validation_filename),
                params.raw_validation_filename,
                int_load_function)
        if params.evaluate or params.attention:
            self.dev_data = ds.DatasetSplit(
                os.path.join(params.data_directory, params.processed_dev_filename),
                params.raw_dev_filename,
                int_load_function)
            if params.enable_testing:
                self.test_data = ds.DatasetSplit(
                    os.path.join(params.data_directory, params.processed_test_filename),
                    params.raw_test_filename,
                    int_load_function)

        def collapse_list(the_list):
            """ Collapses a list of list into a single list."""
            return [s for i in the_list for s in i]

        train_input_seqs = []
        train_input_seqs = collapse_list(
            self.train_data.get_ex_properties(
                lambda i: i.input_seqs()))

        self.input_vocabulary = ATISVocabulary(
            train_input_seqs,
            os.path.join(params.data_directory, params.input_vocabulary_filename),
            params,
            True,
            anonymizer=self.anonymizer if params.anonymization_scoring else None)

        train_output_seqs = collapse_list(
            self.train_data.get_ex_properties(
                lambda i: i.output_seqs()))

        self.output_vocabulary = ATISVocabulary(
            train_output_seqs,
            os.path.join(params.data_directory, params.output_vocabulary_filename),
            params,
            False,
            anonymizer=self.anonymizer if params.anonymization_scoring else None)

    def get_all_utterances(self,
                           dataset,
                           max_input_length=float('inf'),
                           max_output_length=float('inf')):
        """ Returns all utterances in a dataset."""
        items = []
        for interaction in dataset.examples:
            for i, utterance in enumerate(interaction.utterances):
                if utterance.length_valid(max_input_length, max_output_length):
                    items.append(atis_batch.UtteranceItem(interaction, i))
        return items

    def get_all_interactions(self,
                             dataset,
                             max_interaction_length=float('inf'),
                             max_input_length=float('inf'),
                             max_output_length=float('inf'),
                             sorted_by_length=False):

        ints = [
            atis_batch.InteractionItem(
                interaction,
                max_input_length,
                max_output_length,
                self.entities_dictionary,
                max_interaction_length) for interaction in dataset.examples]
        if sorted_by_length:
            return sorted(ints, key=lambda x: len(x))[::-1]
        else:
            return ints

    # This defines a standard way of training: each example is an utterance, and
    # the batch can contain unrelated utterances.
    def get_utterance_batches(self,
                              batch_size,
                              max_input_length=float('inf'),
                              max_output_length=float('inf'),
                              randomize=True):
        # First, get all interactions and the positions of the utterances that are
        # possible in them.
        items = self.get_all_utterances(self.train_data,
                                        max_input_length,
                                        max_output_length)
        if randomize:
            random.shuffle(items)

        batches = []

        current_batch_items = []
        for item in items:
            if len(current_batch_items) >= batch_size:
                batches.append(atis_batch.UtteranceBatch(current_batch_items))
                current_batch_items = []
            current_batch_items.append(item)
        batches.append(atis_batch.UtteranceBatch(current_batch_items))

        assert sum([len(batch) for batch in batches]) == len(items)

        return batches

    def get_interaction_batches(self,
                                batch_size,
                                max_interaction_length=float('inf'),
                                max_input_length=float('inf'),
                                max_output_length=float('inf'),
                                randomize=True):
        items = self.get_all_interactions(self.train_data,
                                          max_interaction_length,
                                          max_input_length,
                                          max_output_length,
                                          sorted_by_length=not randomize)
        if randomize:
            random.shuffle(items)

        batches = []
        current_batch_items = []
        for item in items:
            if len(current_batch_items) >= batch_size:
                batches.append(
                    atis_batch.InteractionBatch(current_batch_items))
                current_batch_items = []
            current_batch_items.append(item)
        batches.append(atis_batch.InteractionBatch(current_batch_items))

        assert sum([len(batch) for batch in batches]) == len(items)

        return batches

    def get_random_utterances(self,
                              num_samples,
                              max_input_length=float('inf'),
                              max_output_length=float('inf')):
        items = self.get_all_utterances(self.train_data,
                                        max_input_length,
                                        max_output_length)
        random.shuffle(items)
        return items[:num_samples]

    def get_random_interactions(self,
                                num_samples,
                                max_interaction_length=float('inf'),
                                max_input_length=float('inf'),
                                max_output_length=float('inf')):
        items = self.get_all_interactions(self.train_data,
                                          max_interaction_length,
                                          max_input_length,
                                          max_output_length)
        random.shuffle(items)
        return items[:num_samples]


def num_utterances(dataset):
    return sum([len(interaction) for interaction in dataset.examples])
