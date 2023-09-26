
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
sampling_method="uniform"



all_frames = [8, 20, 30, 40, 45, 50, 55, 60,
                65, 70, 75, 80, 85,
                90, 95, 103, 113, 121, 133, 140]


def sampling(csv_path,
             sampling_ratio,
             sampling_method):
    """
    Given input csv, returned sampled results

    :param csv_path:
    :param sampling_ratio:
    :param sampling_method: "uniform", "enriched_tail", "long_tail"

    :return:
    """
    # Load distance CSV into data frame
    data = pd.read_csv(csv_path, delimiter=',')
    print(data)

    # return




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
             sampling_ratio=sampling_ratio,
             sampling_method=sampling_method)
