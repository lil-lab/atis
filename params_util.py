import os
import pickle


def check_same_params(param_filename, current_params):
    ofile = open(param_filename)
    loaded_params = pickle.load(ofile)
    ofile.close()

    same = True
    loaded_vars = vars(loaded_params)
    current_vars = vars(current_params)

    for attribute, value in loaded_vars.items():
        if attribute not in current_vars:
            print(
                "Loaded attribute " +
                str(attribute) +
                " not defined in current parameters")
            same = False
        elif value != current_vars[attribute]:
            print("Attribute " +
                  str(attribute) +
                  " has different value in loaded and current parameters (" +
                  str(value) +
                  " vs. " +
                  str(current_vars[attribute]) +
                  ")")
            same = False

    for attribute in current_vars:
        if attribute not in loaded_vars:
            print(
                "Current attribute " +
                str(attribute) +
                " not defined in loaded parameters (filename: " +
                str(param_filename) +
                ")")
            same = False

    assert same, "loaded and current parameters are not the same"


def write_params(param_filename, params):
    if os.path.exists(param_filename):
        check_same_params(param_filename, params)
    else:
        outfile = open(param_filename, "w")
        pickle.dump(params, outfile)
        outfile.close()
