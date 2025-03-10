import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp, pi
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min, weibull_max, genextreme
import itertools



all_frames_uniform = [1, 2, 4, 5, 6, 36, 38, 40, 42, 45,
                      47, 50, 52, 55, 57, 60, 62, 65, 67, 70,
                      72, 74, 76, 78, 80, 84, 86, 87, 100, 142]

all_frames_enriched = [1, 2, 4, 5, 6, 36, 38, 40, 42, 45, 47,
                       50, 52, 55, 57, 60, 62, 65, 67, 70,
                      72, 74, 76, 78, 80, 86, 101, 111, 121, 132]

all_frames_long_tail = [1, 2, 4, 5, 6, 36, 38, 40, 42, 45, 47,
                        50, 52, 55, 57, 60, 62, 65, 67, 70, 72,
                        74, 76, 78, 80, 84, 86, 87, 100, 142]


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))




# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)




def evt_predict(data,
                shape,
                loc,
                scale,
                distance_choice,
                distribution,
                flip_distance):
    """

    :param data:
    :param shape:
    :param loc:
    :param scale:
    :return:
    """
    if distance_choice == "to_A":
        sample_w_frame = data[["frame", "distance_to_A"]]

        if flip_distance:
            max_value = data["distance_to_A"].max()
            sample_w_frame["distance_to_A"] = max_value - sample_w_frame["distance_to_A"]

    else:
        sample_w_frame = data[["frame", "distance_to_B"]]

        if flip_distance:
            max_value = data["distance_to_B"].max()
            sample_w_frame["distance_to_B"] = max_value - sample_w_frame["distance_to_B"]

    frames = sample_w_frame['frame'].unique()

    prob_dict = {}
    prob_one_frame = []

    for one_frame in frames:
        frames = sample_w_frame.loc[sample_w_frame['frame'] == one_frame]

        if distance_choice == "to_A":
            dist = frames["distance_to_A"].tolist()
        else:
            dist = frames["distance_to_B"].tolist()

        for one_dist in dist:
            # Always use CDF for weibull and reverse weibull
            if distribution == "weibull":
                prob = weibull_min.cdf(one_dist, shape, loc, scale)
            elif distribution == "reversed_weibull":
                prob = weibull_max.cdf(one_dist, shape, loc, scale)
            else:
                raise Exception('Distribution not implemented.')

            prob_one_frame.append(prob)

        prob_dict[one_frame] = mean(prob_one_frame)

    return prob_dict




def evt_model(training_a_data_path,
              training_b_data_path,
              test_data_path,
              tail_ratio,
              distribution,
              distance_choice,
              sampling,
              flip_distance,
              params=None,
              fit_curve=False):
    """

    :return:
    """

    # Load training data for class A and B, and testing data
    training_data_a = pd.read_csv(training_a_data_path)
    training_data_b = pd.read_csv(training_b_data_path)
    test_data = pd.read_csv(test_data_path)

    print(training_data_a.shape)
    print(training_data_b.shape)

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

        # EVT: calculate how many samples we will use based on tail size
        nb_sample = int(tail_ratio * training_data_b.shape[0])
        print("nb_sample for B: ", nb_sample)

        # EVT Uniform
        if sampling == "uniform":
            data_for_b = training_data_b[:nb_sample]
            # Use only a part of the data to fit EVT
            prob_dict_b = evt_predict(data=data_for_b,
                                      shape=shape_b,
                                      loc=loc_b,
                                      scale=scale_b,
                                      distribution=distribution,
                                      distance_choice=distance_choice,
                                      flip_distance=flip_distance)


        # EVT enriched tail
        elif sampling == "enrich_tail":
            # Use tail type to decide what samples we are taking
            data_for_b_left = training_data_b[:int(0.5 * nb_sample)]
            data_for_b_right = training_data_b[-int(0.5 * nb_sample):]
            frames = [data_for_b_left, data_for_b_right]
            data_for_b = pd.concat(frames)

            prob_dict_b = evt_predict(data=data_for_b,
                                      shape=shape_b,
                                      loc=loc_b,
                                      scale=scale_b,
                                      distance_choice=distance_choice,
                                      distribution=distribution,
                                      flip_distance=flip_distance)

        # EVT long tail (new)
        elif sampling == "long_tail":
            # First, calculate how many samples we will use based on tail size
            data_for_b = training_data_b[-nb_sample:]

            prob_dict_b = evt_predict(data=data_for_b,
                                      shape=shape_b,
                                      loc=loc_b,
                                      scale=scale_b,
                                      distribution=distribution,
                                      distance_choice=distance_choice,
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



if __name__ == "__main__":
    # Use data that is not shuffled
    class_a_uniform_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                           "sampled_data_0227/EVT/sample_a.csv"
    class_b_uniform_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                           "sampled_data_0227/EVT/sample_b_uniform_use_for_both_ab.csv"

    class_b_enrich_dist_to_a_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                                    "sampled_data_0227/EVT/sample_b_enrich_sort_by_distance_to_a.csv"
    class_b_enrich_dist_to_b_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                                    "sampled_data_0227/EVT/sample_b_enrich_sort_by_distance_to_b.csv"

    class_b_long_dist_to_a_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                                  "sampled_data_0227/EVT/sample_b_long_sort_by_distance_to_a.csv"
    class_b_long_dist_to_b_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                                  "sampled_data_0227/EVT/sample_b_long_sort_by_distance_to_b.csv"

    test_sample_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                       "sampled_data_0227/EVT/test_samples.csv"

    tail_size = 0.5

    # Uniform:
    # param_list = ["reversed_weibull", "to_B", True, [1.0, 0.9, 30]]
    # print(param_list)
    #
    # evt_uniform = evt_model(training_a_data_path=class_a_uniform_path,
    #                           training_b_data_path=class_b_uniform_path,
    #                           test_data_path=test_sample_path,
    #                           tail_ratio=tail_size,
    #                           sampling="uniform",
    #                           distribution=param_list[0],
    #                           distance_choice=param_list[1],
    #                           flip_distance=param_list[2],
    #                           params=param_list[3])
    # print(evt_uniform_inclusion)

    # Long tail
    # param_list = ["reversed_weibull", "to_B", True, [1.0, 0.9, 7]]
    # print(param_list)
    #
    # evt_long_tail = evt_model(training_a_data_path=class_a_uniform_path,
    #                           training_b_data_path=class_b_long_dist_to_b_path,
    #                           test_data_path=test_sample_path,
    #                           tail_ratio=tail_size,
    #                           sampling="long_tail",
    #                           distribution=param_list[0],
    #                           distance_choice=param_list[1],
    #                           flip_distance=param_list[2],
    #                           params=param_list[3])
    # print(evt_long_tail)

    # Enriched tail
    param_list = ["reversed_weibull", "to_B", True, [1.0, 0.9, 30]]
    print(param_list)

    evt_enriched_tail = evt_model(training_a_data_path=class_a_uniform_path,
                              training_b_data_path=class_b_enrich_dist_to_b_path,
                              test_data_path=test_sample_path,
                              tail_ratio=tail_size,
                              sampling="enrich_tail",
                              distribution=param_list[0],
                              distance_choice=param_list[1],
                              flip_distance=param_list[2],
                              params=param_list[3])
    print(evt_enriched_tail)