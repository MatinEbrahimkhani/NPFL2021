#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_count = dict()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")

            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_count[line] = 1 if data_count.get(line) == None else data_count[line] + 1
    print(data_count)
    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_count = list(data_count.values())
    data_dist = np.array(data_count + [0]) / sum(data_count)
    # zero in data_count+[0] indicates the probability of string "D" in the data distribution
    # this would be used in cross entropy

    # print(data_dist)
    # TODO: Load model distribution, each line `string \t probability`.
    model_count = dict()
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            elements = line.split("\t")
            model_count[elements[0]] = float(elements[1])

    # print(model_count)
    # TODO: Create a NumPy array containing the model distribution.
    model_dist = np.array(list(model_count.values()))
    # print(model_dist)
    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    # print("******")
    # print(np.log(data_dist))
    entropy = - np.sum(data_dist[:-1] * np.log(data_dist[:-1]))
    # Eliminating the string "D" distribution from the calculation of entropy
    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.

    crossentropy = - np.sum(data_dist * np.log(model_dist))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
