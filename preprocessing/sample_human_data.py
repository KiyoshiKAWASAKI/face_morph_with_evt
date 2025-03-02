import csv
import os, sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


sampling_ratio = 0.9
tail_weight = 0.3

# sampling_method = "uniform"
sampling_method = "enriched_tail"

save_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data"

# Path and params. Normally, no need to change these
processed_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                     "face_morph_data/processed_morph300.csv"

nb_training = 4 # Do not change it - match with human data
all_frames = [8, 20, 30, 40, 45, 50, 55, 60,
              65, 70, 75, 80, 85,
              90, 95, 103, 113, 121, 133, 140]




def sampling(csv_path,
             frame_index,
             nb_training,
             sampling_ratio,
             sampling_method,
             tail_weight):
    """

    :param csv_path:
    :param frame_index:
    :param nb_training: ideally, choose among 3, 4, and 5
    :param sampling_ratio:
    :param sampling_method: uniform, enriched_tail and long_tail
    :param tail_weight: how much to sample from the tail(s)

    :return: 3 data frames for class A, class B and gap, respectively
    """
    # Load distance CSV into data frame
    data = pd.read_csv(csv_path, delimiter=',')

    # Get frame and worker response
    data_selected = data[["worker_response", "test_img"]]
    data_selected['frame_index'] = data_selected.test_img.str. \
        split('/').str[-1].str.split("_").str[-1].str.split(".").str[0]

    data_final = data_selected[["frame_index", "worker_response"]]
    data_final["frame_index"] = pd.to_numeric(data_final["frame_index"])

    # Find training frames for A and B, and frames in the gap
    training_frame_for_A = frame_index[:nb_training]
    training_frame_for_B = frame_index[-nb_training:]
    testing_frame = list(set(all_frames) - set(training_frame_for_A) - set(training_frame_for_B))

    # Find rows for training and testing
    training_data_A = data_final.loc[data_final["frame_index"].isin(training_frame_for_A)]
    training_data_B = data_final.loc[data_final["frame_index"].isin(training_frame_for_B)]
    testing_data = data_final.loc[data_final["frame_index"].isin(testing_frame)]

    assert training_data_A.shape[0] + training_data_B.shape[0] + testing_data.shape[0] == data_final.shape[0]

    # Uniformly sample each part and then concat
    if sampling_method == "uniform":
        nb_training_sample = int(training_data_A.shape[0] * sampling_ratio)
        nb_testing_sample = int(testing_data.shape[0] * sampling_ratio)

        sample_A = training_data_A.sample(nb_training_sample)
        sample_B = training_data_B.sample(nb_training_sample)
        test_sample = testing_data.sample(nb_testing_sample)

        final_samples = pd.concat([sample_A, test_sample, sample_B], ignore_index=True)

        file_name = "human_" + sampling_method + "_nb_train_" + str(nb_training) + "_ratio_" + \
                     str(sampling_ratio) + ".csv"

        final_samples.to_csv(os.path.join(save_csv_path, file_name))


    elif sampling_method == "enriched_tail":
        # nb_training_sample = int(data_final.shape[0] * sampling_ratio * tail_weight)
        # nb_testing_sample = data_final.shape[0] - 2 * nb_training

        nb_training_sample = int(training_data_A.shape[0] * sampling_ratio)
        nb_testing_sample = int(testing_data.shape[0] * 0.5)

        sample_A = training_data_A.sample(nb_training_sample)
        sample_B = training_data_B.sample(nb_training_sample)
        test_sample = testing_data.sample(nb_testing_sample)

        final_samples = pd.concat([sample_A, test_sample, sample_B], ignore_index=True)

        file_name = "human_" + sampling_method + "_nb_train_" + str(nb_training) + "_ratio_" + \
                     str(sampling_ratio) + "_tail_weight_" + str(tail_weight) + ".csv"

        final_samples.to_csv(os.path.join(save_csv_path, file_name))

    elif sampling_method == "long_tail":
        # TODO: maybe adding long tail later -- it is not in walters proposal
        pass
        # Find number of samples for the long tail.
        # nb_sample_tail = int(nb_training_sample * tail_weight)
        # nb_sample_middle = nb_training_sample - nb_sample_tail
        #
        # # Sort values
        # training_data_B = training_data_B.sort_values('distance_to_A')
        # training_data_B_sampled = training_data_B.sample(nb_training_sample)
        #
        # # Find all samples for each part
        # tail_samples = training_data_B_sampled.tail(nb_sample_tail)
        # middle_samples =training_data_B_sampled[:-nb_sample_tail].sample(nb_sample_middle)
        #
        # # Combine 2 dataframes
        # sample_B = pd.concat([middle_samples, tail_samples], ignore_index=True)

    else:
        raise Exception("Empty sample for class B. Sampling methods supported are: uniform, enriched_tail and long_tail.")


if __name__ == "__main__":
    sampling(csv_path=processed_csv_path,
             frame_index=all_frames,
             nb_training=nb_training,
             sampling_ratio=sampling_ratio,
             sampling_method=sampling_method,
             tail_weight=tail_weight)