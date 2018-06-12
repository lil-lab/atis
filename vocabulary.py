import operator
import os
import pickle

# Special sequencing tokens.
UNK_TOK = "_UNK"  # Replaces out-of-vocabulary words.
EOS_TOK = "_EOS"  # Appended to the end of a sequence to indicate its end.
DEL_TOK = ";"


class Vocabulary:
    def get_vocab(self, sequences, ignore_fn):
        type_counts = {}

        for sequence in sequences:
            for token in sequence:
                if not ignore_fn(token):
                    if token not in type_counts:
                        type_counts[token] = 0
                    type_counts[token] += 1

        # Create sorted list of tokens, by their counts. Reverse so it is in order of
        # most frequent to least frequent.
        sorted_type_counts = sorted(sorted(type_counts.items()),
                                    key=operator.itemgetter(1))[::-1]

        sorted_types = [typecount[0]
                        for typecount in sorted_type_counts if typecount[1] >= self.min_occur]

        # Append the necessary functional tokens.
        sorted_types = self.functional_types + sorted_types

        # Cut off if vocab_size is set (nonnegative)
        if self.max_size >= 0:
            vocab = sorted_types[:max(self.max_size, len(sorted_types))]
        else:
            vocab = sorted_types

        return vocab

    def __init__(self,
                 sequences,
                 filename,
                 functional_types=[],
                 max_size=-1,
                 min_occur=0,
                 ignore_fn=lambda x: False):
        self.functional_types = functional_types
        self.max_size = max_size
        self.min_occur = min_occur

        vocab = self.get_vocab(sequences, ignore_fn)

        self.id_to_token = []
        self.token_to_id = {}

        for i in range(len(vocab)):
            self.id_to_token.append(vocab[i])
            self.token_to_id[vocab[i]] = i

        # Load the previous vocab, if it exists.
        if os.path.exists(filename):
            f = open(filename, 'rb')
            loaded_vocab = pickle.load(f)
            f.close()

            print("Loaded vocabulary from " + str(filename))
            if loaded_vocab.id_to_token != self.id_to_token or loaded_vocab.token_to_id != self.token_to_id:
                print("Loaded vocabulary is different than generated vocabulary.")
        else:
            print("Writing vocabulary to " + str(filename))
            f = open(filename, 'wb')
            pickle.dump(self, f)
            f.close()

    def __len__(self):
        return len(self.id_to_token)
