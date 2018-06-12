"""Gets and stores vocabulary for the ATIS data."""

import snippets
from vocabulary import Vocabulary, UNK_TOK, DEL_TOK, EOS_TOK

INPUT_FN_TYPES = [UNK_TOK, DEL_TOK, EOS_TOK]
OUTPUT_FN_TYPES = [UNK_TOK, EOS_TOK]

MIN_INPUT_OCCUR = 2
MIN_OUTPUT_OCCUR = 1

class ATISVocabulary():
    """ Stores the vocabulary for the ATIS data.

    Attributes:
        raw_vocab (Vocabulary): Vocabulary object.
        tokens (set of str): Set of all of the strings in the vocabulary.
        inorder_tokens (list of str): List of all tokens, with a strict and
            unchanging order.
    """
    def __init__(self,
                 token_sequences,
                 filename,
                 params,
                 is_input,
                 anonymizer=None):
        self.raw_vocab = Vocabulary(
            token_sequences,
            filename,
            functional_types=INPUT_FN_TYPES if is_input else OUTPUT_FN_TYPES,
            min_occur=MIN_INPUT_OCCUR if is_input else MIN_OUTPUT_OCCUR,
            ignore_fn=lambda x: snippets.is_snippet(x) or (
                anonymizer and anonymizer.is_anon_tok(x)))
        self.tokens = set(self.raw_vocab.token_to_id.keys())
        self.inorder_tokens = self.raw_vocab.id_to_token

        assert len(self.inorder_tokens) == len(self.raw_vocab)

    def __len__(self):
        return len(self.raw_vocab)

    def token_to_id(self, token):
        """ Maps from a token to a unique ID.

        Inputs:
            token (str): The token to look up.

        Returns:
            int, uniquely identifying the token.
        """
        return self.raw_vocab.token_to_id[token]

    def id_to_token(self, identifier):
        """ Maps from a unique integer to an identifier.

        Inputs:
            identifier (int): The unique ID.

        Returns:
            string, representing the token.
        """
        return self.raw_vocab.id_to_token[identifier]
