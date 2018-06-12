import copy
import json
import util

ENTITY_NAME = "ENTITY"
CONSTANT_NAME = "CONSTANT"
TIME_NAME = "TIME"
SEPARATOR = "#"


def timeval(s):
    if s.endswith("am") or s.endswith("pm") and s[:-2].isdigit():
        numval = int(s[:-2])
        if len(s) == 3 or len(s) == 4:
            numval *= 100
        if s.endswith("pm"):
            numval += 1200
        return str(numval)
    return ""


def is_time(s):
    if s.endswith("am") or s.endswith("pm"):
        if s[:-2].isdigit():
            return True

    return False


def deanonymize(sequence, ent_dict, key):
    new_sequence = []
    for token in sequence:
        if token in ent_dict:
            new_sequence.extend(ent_dict[token][key])
        else:
            new_sequence.append(token)

    return new_sequence


class Anonymizer:
    def __init__(self, filename):
        self.anonymization_map = []
        self.entity_types = []
        self.keys = set()

        pairs = [json.loads(line) for line in open(filename).readlines()]
        for pair in pairs:
            for key in pair:
                if key != "type":
                    self.keys.add(key)
            self.anonymization_map.append(pair)
            if not pair["type"] in self.entity_types:
                self.entity_types.append(pair["type"])

        self.entity_types.append(ENTITY_NAME)
        self.entity_types.append(CONSTANT_NAME)
        self.entity_types.append(TIME_NAME)

        self.entity_set = set(self.entity_types)

    def get_entity_type_from_token(self, token):
        # these are in the pattern NAME:#, so just strip the thing after the
        # colon
        colon_loc = token.index(SEPARATOR)
        entity_type = token[:colon_loc]
        assert entity_type in self.entity_set

        return entity_type

    def is_anon_tok(self, token):
        return token.split(SEPARATOR)[0] in self.entity_set

    def get_anon_id(self, token):
        if self.is_anon_tok(token):
            return self.entity_types.index(token.split(SEPARATOR)[0])
        else:
            return -1

    def extract_dates(self, anonymized_sequence, key, type_counts, tok_to_ent_dict):
        all_values = [val[key][0] for val in self.anonymization_map]

        years = [val[key][0] for val in self.anonymization_map if val["type"] == "YEAR"]
        months = [val[key][0] for val in self.anonymization_map if val["type"] == "MONTH_NUMBER"]
        dates = [val[key] for val in self.anonymization_map if val["type"] == "DAY_NUMBER"]

        new_seq = []
        # First detect dates: (year month day)
        i = 0
        while i < len(anonymized_sequence):
            if i + 2 < len(anonymized_sequence) and anonymized_sequence[i] in years and \
                anonymized_sequence[i+1] in months and \
                [anonymized_sequence[i+2]] in dates:

                year = anonymized_sequence[i]
                month = anonymized_sequence[i+1]

                month_number = ""
                for pair in self.anonymization_map:
                    if pair[key] == [month]:
                        month_number = pair["sql"][0]
                        break

                new_token = DATE_NAME + SEPARATOR + str(type_counts[DATE_NAME])
                new_pair = {"type": DATE_NAME}

                # Also handle the case where the day has two tokens
                if i + 3 < len(anonymized_sequence) and [anonymized_sequence[i+2],
                    anonymized_sequence[i+3]] in dates:
                    day = [anonymized_sequence[i+2], anonymized_sequence[i+3]]
                    new_pair[key] = anonymized_sequence[i:i+4]
                    i += 4
                else:
                    day = [anonymized_sequence[i+2]]
                    new_pair[key] = anonymized_sequence[i:i+3]
                    i += 3

                # check that the date isn't already in the map
                found_key = ""
                for name, pair in tok_to_ent_dict.items():
                    if pair[key] == new_pair[key]:
                        found_key = name

                if found_key:
                    new_seq.append(found_key)
                else:
                    day_number = ""
                    for pair in self.anonymization_map:
                        if pair[key] == day:
                            day_number = pair["sql"]
                            break
                    assert day_number, "Could not find " + str(day) + " in anonymization map"
                    new_pair["sql"] = ["date_day.year",
                                       "=",
                                       year,
                                       "AND",
                                       "date_day.month_number",
                                       "=",
                                       month_number,
                                       "AND",
                                       "date_day.day_number",
                                       "=",
                                       day_number[0]]

                    new_seq.append(new_token)
                    type_counts[DATE_NAME] += 1

                    tok_to_ent_dict[new_token] = new_pair
            else:
                new_seq.append(anonymized_sequence[i])
                i += 1

        return new_seq

    def anonymize(self,
                  sequence,
                  tok_to_entity_dict,
                  key,
                  add_new_anon_toks=False):
        # Sort the token-tok-entity dict by the length of the modality.
        sorted_dict = sorted(tok_to_entity_dict.items(),
                             key=lambda k: len(k[1][key]))[::-1]

        anonymized_sequence = copy.deepcopy(sequence)



        if add_new_anon_toks:
            type_counts = {}
            for entity_type in self.entity_types:
                type_counts[entity_type] = 0
            for token in tok_to_entity_dict:
                entity_type = self.get_entity_type_from_token(token)
                type_counts[entity_type] += 1

        # First find occurrences of things in the anonymization dictionary.
        for token, modalities in sorted_dict:
            our_modality = modalities[key]

            # Check if this key's version of the anonymized thing is in our
            # sequence.
            while util.subsequence(our_modality, anonymized_sequence):
                found = False
                for startidx in range(len(anonymized_sequence) - len(our_modality) + 1):
                    if anonymized_sequence[startidx:startidx +
                                           len(our_modality)] == our_modality:
                        anonymized_sequence = anonymized_sequence[:startidx] + [
                            token] + anonymized_sequence[startidx + len(our_modality):]
                        found = True
                        break
                assert found, "Thought " + \
                    str(our_modality) + " was in [" + str(anonymized_sequence) + "] but could not find it"


        # Now add new keys if they are present.
        if add_new_anon_toks:

            # For every span in the sequence, check whether it is in the anon map
            # for this modality
            sorted_anon_map = sorted(self.anonymization_map,
                                     key=lambda k: len(k[key]))[::-1]

            for pair in sorted_anon_map:
                our_modality = pair[key]

                token_type = pair["type"]
                new_token = token_type + SEPARATOR + \
                    str(type_counts[token_type])

                while util.subsequence(our_modality, anonymized_sequence):
                    found = False
                    for startidx in range(len(anonymized_sequence) - len(our_modality) + 1):
                        if anonymized_sequence[startidx:startidx + \
                            len(our_modality)] == our_modality:
                            if new_token not in tok_to_entity_dict:
                                type_counts[token_type] += 1
                                tok_to_entity_dict[new_token] = pair

                            anonymized_sequence = anonymized_sequence[:startidx] + [
                                new_token] + anonymized_sequence[startidx + len(our_modality):]
                            found = True
                            break
                    assert found, "Thought " + \
                        str(our_modality) + " was in [" + str(anonymized_sequence) + "] but could not find it"

            # Also replace integers with constants
            for index, token in enumerate(anonymized_sequence):
                if token.isdigit() or is_time(token):
                    if token.isdigit():
                        entity_type = CONSTANT_NAME
                        value = new_token
                    if is_time(token):
                        entity_type = TIME_NAME
                        value = timeval(token)

                    # First try to find the constant in the entity dictionary already,
                    # and get the name if it's found.
                    new_token = ""
                    new_dict = {}
                    found = False
                    for entity, value in tok_to_entity_dict.items():
                        if value[key][0] == token:
                            new_token = entity
                            new_dict = value
                            found = True
                            break

                    if not found:
                        new_token = entity_type + SEPARATOR + \
                            str(type_counts[entity_type])
                        new_dict = {}
                        for tempkey in self.keys:
                            new_dict[tempkey] = [token]

                        tok_to_entity_dict[new_token] = new_dict
                        type_counts[entity_type] += 1

                    anonymized_sequence[index] = new_token

        return anonymized_sequence
