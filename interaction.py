""" Contains the class for an interaction in ATIS. """

import anonymization as anon
import sql_util
from snippets import expand_snippets
from utterance import Utterance, OUTPUT_KEY, ANON_INPUT_KEY

class Interaction:
    """ ATIS interaction class.

    Attributes:
        utterances (list of Utterance): The utterances in the interaction.
        snippets (list of Snippet): The snippets that appear through the interaction.
        anon_tok_to_ent:
        identifier (str): Unique identifier for the interaction in the dataset.
    """
    def __init__(self,
                 utterances,
                 snippets,
                 anon_tok_to_ent,
                 identifier,
                 params):
        self.utterances = utterances
        self.snippets = snippets
        self.anon_tok_to_ent = anon_tok_to_ent
        self.identifier = identifier

        # Ensure that each utterance's input and output sequences, when remapped
        # without anonymization or snippets, are the same as the original
        # version.
        for i, utterance in enumerate(self.utterances):
            deanon_input = self.deanonymize(utterance.input_seq_to_use,
                                            ANON_INPUT_KEY)
            assert deanon_input == utterance.original_input_seq, "Anonymized sequence [" \
                + " ".join(utterance.input_seq_to_use) + "] is not the same as [" \
                + " ".join(utterance.original_input_seq) + "] when deanonymized (is [" \
                + " ".join(deanon_input) + "] instead)"
            desnippet_gold = self.expand_snippets(utterance.gold_query_to_use)
            deanon_gold = self.deanonymize(desnippet_gold, OUTPUT_KEY)
            assert deanon_gold == utterance.original_gold_query, \
                "Anonymized and/or snippet'd query " \
                + " ".join(utterance.gold_query_to_use) + " is not the same as " \
                + " ".join(utterance.original_gold_query)

    def __str__(self):
        string = "Utterances:\n"
        for utterance in self.utterances:
            string += str(utterance) + "\n"
        string += "Anonymization dictionary:\n"
        for ent_tok, deanon in self.anon_tok_to_ent.items():
            string += ent_tok + "\t" + str(deanon) + "\n"

        return string

    def __len__(self):
        return len(self.utterances)

    def deanonymize(self, sequence, key):
        """ Deanonymizes a predicted query or an input utterance.

        Inputs:
            sequence (list of str): The sequence to deanonymize.
            key (str): The key in the anonymization table, e.g. NL or SQL.
        """
        return anon.deanonymize(sequence, self.anon_tok_to_ent, key)

    def expand_snippets(self, sequence):
        """ Expands snippets for a sequence.

        Inputs:
            sequence (list of str): A SQL query.

        """
        return expand_snippets(sequence, self.snippets)

    def input_seqs(self):
        in_seqs = []
        for utterance in self.utterances:
            in_seqs.append(utterance.input_seq_to_use)
        return in_seqs

    def output_seqs(self):
        out_seqs = []
        for utterance in self.utterances:
            out_seqs.append(utterance.gold_query_to_use)
        return out_seqs

# raw_load_function


def load_function(parameters,
                  nl_to_sql_dict,
                  anonymizer):
    def fn(interaction_example):
        keep = False

        raw_utterances = interaction_example["interaction"]
        identifier = interaction_example["id"]

        snippet_bank = []

        utterance_examples = []

        anon_tok_to_ent = {}

        for utterance in raw_utterances:
            available_snippets = [
                snippet for snippet in snippet_bank if snippet.index <= 1]

            proc_utterance = Utterance(utterance,
                                       available_snippets,
                                       nl_to_sql_dict,
                                       parameters,
                                       anon_tok_to_ent,
                                       anonymizer)
            keep_utterance = proc_utterance.keep

            if keep_utterance:
                keep = True
                utterance_examples.append(proc_utterance)

                # Update the snippet bank, and age each snippet in it.
                if parameters.use_snippets:
                    snippets = sql_util.get_subtrees(
                        proc_utterance.anonymized_gold_query,
                        proc_utterance.available_snippets)

                    for snippet in snippets:
                        snippet.assign_id(len(snippet_bank))
                        snippet_bank.append(snippet)

                for snippet in snippet_bank:
                    snippet.increase_age()

        interaction = Interaction(utterance_examples,
                                  snippet_bank,
                                  anon_tok_to_ent,
                                  identifier,
                                  parameters)

        return interaction, keep

    return fn
