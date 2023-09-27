
import sys, os
import numpy as np
import matplotlib as plt
import pandas as pd



# Paths for result files
result_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/deepface.csv"
model_name = "deepface"


# Set number of training frames as a parameter
nb_training = 3
sampling_ratio = 1.0
sampling_method = "uniform"



all_frames = [8, 20, 30, 40, 45, 50, 55, 60,
                65, 70, 75, 80, 85,
                90, 95, 103, 113, 121, 133, 140]


def sampling(csv_path,
             frame_index,
             nb_training,
             sampling_ratio,
             sampling_method,
             tail_weight=0.4):
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

    # TODO: might need to add extra steps for Deeksha's data - JH

    # Find training frames for A and B, and frames in the gap
    training_frame_for_A = frame_index[:nb_training]
    training_frame_for_B = frame_index[-nb_training:]
    testing_frame = list(set(all_frames) - set(training_frame_for_A) - set(training_frame_for_B))

    # Reset column names
    data = data[["0", "1", "2", "3"]]
    data.columns = ['morph_name', 'frame', 'distance_to_A', 'distance_to_B']

    # Find rows for training and testing
    training_data_A = data.loc[data['frame'].isin(training_frame_for_A)]
    training_data_B = data.loc[data['frame'].isin(training_frame_for_B)]
    testing_data = data.loc[data['frame'].isin(testing_frame)]

    assert training_data_A.shape[0] + training_data_B.shape[0] + testing_data.shape[0] == data.shape[0]

    """
    Sampling class A:
        Calculate number of samples
        Shuffle the data and take sample
        Note: Always use the same distribution
    """
    nb_training_sample = int(training_data_A.shape[0] * sampling_ratio)
    sample_A = training_data_A.sample(nb_training_sample)

    # For class B, it differs for each sampling method
    if sampling_method == "uniform":
        sample_B = training_data_B.sample(nb_training_sample)

    elif sampling_method == "enriched_tail":
        # Find number of samples for the enriched tails.
        nb_sample_head = int(nb_training_sample * tail_weight)
        nb_sample_tail = nb_sample_head
        nb_sample_middle = nb_training_sample - nb_sample_head - nb_sample_tail

        # According to the frame indices, get samples for each part
        head_index = training_frame_for_B[0]
        tail_index = training_frame_for_B[-1]
        middle_index = list(set(training_frame_for_B) - set(head_index) - set(tail_index))

        # Find all samples for each part
        head_samples = training_data_B.loc[training_data_B['frame'].isin(head_index)]
        tail_samples = training_data_B.loc[training_data_B['frame'].isin(tail_index)]
        middle_samples = training_data_B.loc[training_data_B['frame'].isin(middle_index)]

        # Then do the sampling
        head_samples = head_samples.sample(nb_sample_head)
        tail_samples = tail_samples.sample(nb_sample_tail)
        middle_samples = middle_samples.sample(nb_sample_middle)

        # Combine 3 dataframes
        sample_B = pd.concat([head_samples, middle_samples, tail_samples], ignore_index=True)

    elif sampling_method == "long_tail":
        # Find number of samples for the long tail.
        nb_sample_tail = int(nb_training_sample * tail_weight * 2)
        nb_sample_middle = nb_training_sample - nb_sample_tail

        # According to the frame indices, get samples for each part
        tail_index = training_frame_for_B[-1]
        middle_index = list(set(training_frame_for_B) - set(tail_index))

        # Find all samples for each part
        tail_samples = training_data_B.loc[training_data_B['frame'].isin(tail_index)]
        middle_samples = training_data_B.loc[training_data_B['frame'].isin(middle_index)]

        # Then do the sampling
        tail_samples = tail_samples.sample(nb_sample_tail)
        middle_samples = middle_samples.sample(nb_sample_middle)

        # Combine 2 dataframes
        sample_B = pd.concat([middle_samples, tail_samples], ignore_index=True)

    else:
        sample_B = None

    # TODO: Testing samples - use all of them? -JH

    if sample_B is not None:
        return sample_A, sample_B, testing_data

    else:
        raise Exception("Empty sample for class B. Methods supported are: uniform, enriched_tail and long_tail")




def modeling(samples,
             model):
    """

    :param samples:
    :param model:
    :return:
    """
    return




if __name__ == "__main__":
    sampling(csv_path=result_csv_path,
             frame_index=all_frames,
             nb_training=nb_training,
             sampling_ratio=sampling_ratio,
             sampling_method=sampling_method)
