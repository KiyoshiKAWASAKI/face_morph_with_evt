import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp, pi
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min, weibull_max, genextreme
import itertools
import warnings
warnings.filterwarnings('ignore')

nb_training = 5

all_frames_uniform = [110, 120, 130, 140, 150, 167, 183, 200, 216, 233, 250, 267, 283, 300, 375, 450, 525, 600]
all_frames_enriched = [110, 120, 130, 140, 150, 167, 183, 200, 216, 233, 250, 267, 283, 300, 325, 350, 575, 600]
all_frames_long = [110, 120, 130, 140, 150, 167, 183, 200, 216, 233, 250, 267, 283, 560, 570, 580, 590, 600]

human_test_ind = [167, 183, 200, 216, 233, 250, 267, 283]

human_uniform_prob = [0.08, 0.04, 0.04, 0.20, 0.52, 0.72, 0.84, 0.96]
human_enrich_prob = [0.00, 0.00, 0.00, 0.08, 0.36, 0.68, 0.84, 0.96]
human_long_prob = [0.04, 0.00, 0.00, 0.00, 0.08, 0.12, 0.16, 0.12]

tail_size = 2

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


def evt_predict(data,
                shape,
                loc,
                scale,
                distribution,
                flip_distance):
    """

    :param data:
    :param shape:
    :param loc:
    :param scale:
    :return:
    """
    if flip_distance:
        max_value = data.max()
        # Calculate the flipped data
        sample_w_frame["distance_to_A"] = max_value - sample_w_frame["distance_to_A"]

    prob_dict = []

    for one_data_point in data:
        # Always use CDF for weibull and reverse weibull
        if distribution == "weibull":
            prob = weibull_min.cdf(one_data_point, shape, loc, scale)

        elif distribution == "reversed_weibull":
            prob = weibull_max.cdf(one_data_point, shape, loc, scale)

        else:
            raise Exception('Distribution not implemented.')

        prob_dict.append(prob)

    return prob_dict


def evt_model(training_a_data,
              training_b_data,
              test_data,
              tail_size,
              distribution,
              sampling,
              flip_distance,
              params=None,
              fit_curve=False):
    """

    :return:
    """


    """
    We can choose to fit a distribution, currently choose between "weibull" and "reversed_weibull".
    This can give estimates on the parameters: fit func returns shape, location and scale parameters.
    But we can also do grid search and tune these parameters instead.
    Currently we choose to do parameter search.

    weibull: weibull_min. When data is bounded from below.
    reversed weibull:weibull_max. When data is bounded from above.
    """
    if fit_curve:
        # We are not using this currently.
        pass

    else:
        """
        Use the parameters for our testing samples.
        Probability for each sample is produced by CDF func.
        """
        shape_b = params[0]
        scale_b = params[1]
        loc_b = params[2]

        prob_dict_a = evt_predict(data=training_data_a,
                                  shape=shape_b,
                                  loc=loc_b,
                                  scale=scale_b,
                                  distribution=distribution,
                                  distance_choice=distance_choice,
                                  flip_distance=flip_distance)


        # Take "tail" samples
        data_for_b = training_data_b[:tail_size]


        # EVT Uniform
        if sampling == "uniform":
            # Use only a part of the data to fit EVT
            prob_dict_b = evt_predict(data=data_for_b,
                                      shape=shape_b,
                                      loc=loc_b,
                                      scale=scale_b,
                                      distribution=distribution,
                                      flip_distance=flip_distance)


        # EVT enriched tail
        elif sampling == "enrich_tail":
            prob_dict_b = evt_predict(data=data_for_b,
                                      shape=shape_b,
                                      loc=loc_b,
                                      scale=scale_b,
                                      distribution=distribution,
                                      flip_distance=flip_distance)

        # EVT long tail (new)
        elif sampling == "long_tail":
            prob_dict_b = evt_predict(data=data_for_b,
                                      shape=shape_b,
                                      loc=loc_b,
                                      scale=scale_b,
                                      distribution=distribution,
                                      flip_distance=flip_distance)

        else:
            print("Invalid sampling method.")

        prob_dict_test = evt_predict(data=test_data,
                                     shape=shape_b,
                                     loc=loc_b,
                                     scale=scale_b,
                                     distribution=distribution,
                                     distance_choice=distance_choice,
                                     flip_distance=flip_distance)

        prob_dict = {**prob_dict_a, **prob_dict_b, **prob_dict_test}
        prob_dict = dict(sorted(prob_dict.items()))

    return prob_dict


def evt_final_prob(inclusion_prob,
                   exclusion_prob):
    """
    inclusion_prob: dictionary
    exclusion_prob: dictionary
    """
    result = {}

    # Iterate through the keys of the first dictionary
    for key in inclusion_prob:
        # Check if the key exists in both dictionaries
        if key in exclusion_prob:
            # Multiply the values and store them in the result dictionary
            result[key] = inclusion_prob[key] * exclusion_prob[key]

    result = {int(k): float(v) for k, v in result.items()}

    return result


if __name__ == "__main__":

    uniform_inclusion_param_list = ["reversed_weibull", "to_B", True, [0.9, 1.0, 15]]
    uniform_inclusion = evt_model(training_a_data_path=class_a_uniform_path,
                                      training_b_data_path=class_b_uniform_dist_to_b_path,
                                      test_data_path=test_sample_path,
                                      tail_ratio=tail_size,
                                      sampling="uniform",
                                      distribution=uniform_inclusion_param_list[0],
                                      distance_choice=uniform_inclusion_param_list[1],
                                      flip_distance=uniform_inclusion_param_list[2],
                                      params=uniform_inclusion_param_list[3])
    evt_uniform_inclusion = {int(k): float(v) for k, v in uniform_inclusion.items()}

    print(evt_uniform_inclusion)