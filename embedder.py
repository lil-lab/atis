""" Embedder for tokens. """

import dynet as dy
import snippets as snippet_handler
import vocabulary as vocabulary_handler


class Embedder():
    """ Embeds tokens. """
    def __init__(self,
                 model,
                 embedding_size,
                 name="",
                 initializer=dy.UniformInitializer(0.1),
                 vocabulary=None,
                 num_tokens=-1,
                 anonymizer=None):
        if vocabulary:
            assert num_tokens < 0, "Specified a vocabulary but also set number of tokens to " + \
                str(num_tokens)
            self.in_vocabulary = lambda token: token in vocabulary.tokens
            self.vocab_token_lookup = lambda token: vocabulary.token_to_id(token)
            self.unknown_token_id = vocabulary.token_to_id(
                vocabulary_handler.UNK_TOK)
            self.vocabulary_size = len(vocabulary)
        else:
            def check_vocab(index):
                """ Makes sure the index is in the vocabulary."""
                assert index < num_tokens, "Passed token ID " + \
                    str(index) + "; expecting something less than " + str(num_tokens)
                return index < num_tokens
            self.in_vocabulary = check_vocab
            self.vocab_token_lookup = lambda x: x
            self.unknown_token_id = num_tokens  # Deliberately throws an error here,
            # But should crash before this
            self.vocabulary_size = num_tokens

        self.anonymizer = anonymizer

        emb_name = name + "-tokens"
        print("Creating token embedder called " + emb_name + " of size " +
              str(self.vocabulary_size) + " x " + str(embedding_size))
        self.token_embedding_matrix = model.add_lookup_parameters(
            (self.vocabulary_size, embedding_size), init=initializer, name=emb_name)

        if self.anonymizer:
            emb_name = name + "-entities"
            entity_size = len(self.anonymizer.entity_types)
            print(
                "Creating entity embedder called " +
                emb_name +
                " of size " +
                str(entity_size) +
                " x " +
                str(embedding_size))
            self.entity_embedding_matrix = model.add_lookup_parameters(
                (entity_size, embedding_size), init=initializer, name=emb_name)

    def __call__(self, token):
        assert isinstance(token, int) or not snippet_handler.is_snippet(token), \
            "embedder should only be called on flat tokens; use snippet_bow if " \
            + "you are trying to encode snippets"

        if self.in_vocabulary(token):
            return self.token_embedding_matrix[self.vocab_token_lookup(token)]
        elif self.anonymizer and self.anonymizer.is_anon_tok(token):
            return self.entity_embedding_matrix[self.anonymizer.get_anon_id(
                token)]
        else:
            return self.token_embedding_matrix[self.unknown_token_id]

    def bow_snippets(self, token, snippets=None):
        """ Bag of words embedding for snippets"""
        if snippet_handler.is_snippet(token):
            assert snippets
            snippet_sequence = []
            for snippet in snippets:
                if snippet.name == token:
                    snippet_sequence = snippet.sequence
                    break
            assert snippet_sequence

            snippet_embeddings = [self(subtoken)
                                  for subtoken in snippet_sequence]

            return dy.average(snippet_embeddings)
        else:
            return self(token)
