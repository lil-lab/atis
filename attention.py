"""Contains classes for computing and keeping track of attention distributions.
"""
from collections import namedtuple

import dynet as dy
import dynet_utils as du

class AttentionResult(namedtuple('AttentionResult',
                                 ('scores',
                                  'distribution',
                                  'vector'))):
    """Stores the result of an attention calculation."""
    __slots__ = ()


class Attention():
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (bool): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (bool): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (bool): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (int): The size of the key vectors.
       value_size (int): The size of the value vectors.
           the query or key.
       query_weights (dy.Parameters): Weights for transforming the query.
       key_weights (dy.Parameters): Weights for transforming the key.
       value_weights (dy.Parameters): Weights for transforming the value.
    """
    def __init__(self,
                 model,
                 query_size,
                 key_size,
                 value_size):
        self.key_size = key_size
        self.value_size = value_size

        self.query_weights = du.add_params(
            model, (query_size, self.key_size), "weights-attention-q")

    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
        assert len(keys) == len(values)
        all_keys = dy.concatenate(keys, d=1)
        all_values = dy.concatenate(values, d=1)

        assert all_keys.dim()[0][0] == self.key_size, "Expected key size of " + \
            str(self.key_size) + " but got " + str(all_keys.dim()[0][0])
        assert all_values.dim()[0][0] == self.value_size

        query = du.linear_transform(query, self.query_weights)

        if du.is_vector(query):
            query = du.add_dim(query)

        return query, all_keys, all_values

    def __call__(self, query, keys, values=None):
        if not values:
            values = keys

        query_t, keys_t, values_t = self.transform_arguments(query,
                                                             keys,
                                                             values)

        scores = dy.transpose(query_t * keys_t)
        distribution = dy.softmax(scores)
        context_vector = values_t * distribution

        return AttentionResult(scores, distribution, context_vector)
