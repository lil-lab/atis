python3 run.py --raw_train_filename="../atis_data/data/resplit/processed/train_with_tables.pkl" \
               --raw_dev_filename="../atis_data/data/resplit/processed/dev_with_tables.pkl" \
               --raw_validation_filename="../atis_data/data/resplit/processed/valid_with_tables.pkl" \
               --raw_test_filename="../atis_data/data/resplit/processed/test_with_tables.pkl" \
               --input_key="utterance" \
               --anonymize=1 \
               --anonymization_scoring=1 \
               --use_snippets=1 \
               --state_positional_embeddings=1 \
               --snippet_age_embedding=1 \
               --discourse_level_lstm=1 \
               --interaction_level=1 \
               --reweight_batch=1 \
               --train=1 

# Explanation of parameters (S# indicates where it's described in the paper):
# anonymize: whether to apply anonymization preprocessing to the input/output pairs (S6).
# anonymization_scoring: whether to score anonymized tokens separately from regular output tokens using attention (S6).
# use_snippets: whether to use the query segment copier (S4.4).
# state_positional_embeddings: whether to add positional embeddings to the input sequence hidden states during attention computations (S4.3)
# snippet_age_embedding: whether to use age embeddings for the segments that are copied (S4.4)
# discourse_level_lstm: whether to use the turn-level encoder (S4.4)
# interaction_level: whether to construct a computation graph for the entire interaction (S4.3; should be set to the same value as discourse_level_lstm)
# reweight_batch: whether to reweight the gradients based on interaction length (S5)

