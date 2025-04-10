import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp, pi
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min, weibull_max, genextreme
import itertools

nb_training = 5

all_frames_uniform = [110, 120, 130, 140, 150, 167, 183, 200, 216, 233, 250, 267, 283, 300, 375, 450, 525, 600]
all_frames_enriched = [110, 120, 130, 140, 150, 167, 183, 200, 216, 233, 250, 267, 283, 300, 325, 350, 575, 600]
all_frames_long = [110, 120, 130, 140, 150, 167, 183, 200, 216, 233, 250, 267, 283, 560, 570, 580, 590, 600]

human_test_ind = [167, 183, 200, 216, 233, 250, 267, 283]

human_uniform_prob = [0.08, 0.04, 0.04, 0.20, 0.52, 0.72, 0.84, 0.96]
human_enrich_prob = [0.00, 0.00, 0.00, 0.08, 0.36, 0.68, 0.84, 0.96]
human_long_prob = [0.04, 0.00, 0.00, 0.00, 0.08, 0.12, 0.16, 0.12]

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

def get_prob(data,
             classifier):
    """

    :param data: just the length of line segments
    :return:
    """
    prob_dict = {}

    for one_data_point in data:
        prob = classifier.predict_proba(np.asarray(data).reshape(-1, 1))
        prob_dict[one_data_point] = mean(prob)

    return prob_dict


def gaussian_naive_bayes(training_a,
                         training_b,
                         testing):
    """
    Use class A and class B training data to fit a gaussian model,
    then use bayesian theorem to calculate probability.

    :return:
    """

    a_sample = training_a
    b_sample = training_b

    print("Number of samples in class A:", len(a_sample))
    print("Number of samples in class B:", len(b_sample))

    # Step 2: Calculate mean and standard deviation
    mean_a = mean(a_sample)
    sd_a = stdev(a_sample)
    var_a = np.var(np.asarray(a_sample))

    mean_b = mean(b_sample)
    sd_b = stdev(b_sample)
    var_b = np.var(np.asarray(b_sample))

    # print("Mean for A: ", mean_a)
    # print("SD for A: ", sd_a)
    # print("Var for A: ", var_a )
    # print("Mean for B: ", mean_b)
    # print("SD for B :", sd_b)
    # print("Var for B: ", var_b)

    stats = [mean_a, sd_a, var_a, mean_b, sd_b, var_b]

    # Build and fit Gaussian Naive Bayes Model
    clf = GaussianNB()

    X = np.asarray(a_sample + b_sample)
    Y = np.asarray([0] * len(a_sample) + [1] * len(b_sample))
    clf.fit(X.reshape(-1, 1), Y)

    # Training data A
    prob_dict_a = get_prob(data=training_a,
                            classifier=clf)

    # Training data B
    prob_dict_b = get_prob(data=training_b,
                           classifier=clf)

    # Testing data
    prob_dict_test = get_prob(data=testing,
                         classifier=clf)

    # Combine dictionaries and sort
    prob_dict = {**prob_dict_a, **prob_dict_b, **prob_dict_test}
    prob_dict = dict(sorted(prob_dict.items()))

    # Take probability for B
    prob_b_list = []

    for one_key in prob_dict.keys():
        prob_list = prob_dict[one_key].tolist()
        prob_b_list.append(prob_list[1])

    return prob_b_list


if __name__ == "__main__":
    gaussian_prob_uniform = gaussian_naive_bayes(training_a=all_frames_uniform[:5],
                                                                training_b=all_frames_uniform[-5:],
                                                                testing=human_test_ind)
    print(gaussian_prob_uniform)

    gaussian_prob_enrich = gaussian_naive_bayes(training_a=all_frames_enriched[:5],
                                                                training_b=all_frames_enriched[-5:],
                                                                testing=human_test_ind)
    print(gaussian_prob_enrich)

    gaussian_prob_long = gaussian_naive_bayes(training_a=all_frames_long[:5],
                                                                training_b=all_frames_long[-5:],
                                                                testing=human_test_ind)
    print(gaussian_prob_long)