import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp, pi
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min, weibull_max, genextreme
from matplotlib.pyplot import figure
import itertools

#####################################################
# Switch to do this on my local laptop
#####################################################
sampling_ratio = 0.9

# TODO: VGG-Face ResNet
model_name = "vggface_resnet"
uniform_csv_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_dist/vgg_resnet_uniform.csv"
enrich_csv_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_dist/vgg_resnet_enriched_tail.csv"
long_tail_csv_path = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_dist/vgg_resnet_long_tail.csv"


data_dir = "/Users/kiyoshi/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/"

#####################################################
# Fixed params
#####################################################
all_frames_uniform = [1, 2, 4, 5, 6,
                      36, 38, 40, 42, 45,
                      47, 50, 52, 55, 57,
                      60, 62, 65, 67, 70,
                      72, 74, 76, 78, 80,
                      86, 101, 111, 121, 132]

all_frames_enriched = [1, 2, 4, 5, 6,
                      36, 38, 40, 42, 45,
                      47, 50, 52, 55, 57,
                      60, 62, 65, 67, 70,
                      72, 74, 76, 78, 80,
                      84, 86, 87, 100, 142]

all_frames_long = [1, 2, 4, 5, 6,
                    36, 38, 40, 42, 45,
                    47, 50, 52, 55, 57,
                    60, 62, 65, 67, 70,
                    72, 74, 76, 78, 80,
                    140, 141, 142]

human_test_ind = [36, 38, 40, 42, 45,
                  47, 50, 52, 55, 57,
                  60, 62, 65, 67, 70,
                  72, 74, 76, 78, 80,]

human_uniform_prob = [0.074786, 0.057692, 0.066239, 0.104701, 0.123932,
                      0.188034, 0.258547, 0.388889, 0.485043, 0.602564,
                      0.690171, 0.782051, 0.856838, 0.884615, 0.899573,
                      0.931624, 0.938034, 0.955128, 0.950855, 0.972222]
human_enrich_prob = [0.119048, 0.062907, 0.075922, 0.097614, 0.132321,
                     0.201735, 0.288503, 0.392625, 0.496746, 0.613883,
                     0.722343, 0.800434, 0.872017, 0.924078, 0.952278,
                     0.952278, 0.958696, 0.976087, 0.973913, 0.980435]

human_long_prob = [0.06572769953051644, 0.018823529411764704, 0.009411764705882352,
                   0.011764705882352941, 0.02358490566037736, 0.030660377358490566,
                   0.04245283018867924, 0.0589622641509434, 0.09669811320754718,
                   0.12028301886792453, 0.18632075471698112, 0.2169811320754717,
                   0.2783018867924528, 0.33490566037735847, 0.42924528301886794,
                   0.5047169811320755, 0.5825471698113207, 0.6279620853080569,
                   0.7156398104265402, 0.7535545023696683]




def sampling(csv_path,
             frame_index,
             nb_training,
             sampling_ratio,
             sampling_method,
             model_name,
             save_sample_path,
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

    # Find training frames for A and B, and frames in the gap
    training_frame_for_A = frame_index[:nb_training]

    # print(model_name)

    if sampling_method != "long_tail":
        training_frame_for_B = frame_index[-nb_training:]
    else:
        training_frame_for_B = frame_index[-3:]

    testing_frame = list(set(frame_index) - set(training_frame_for_A) - set(training_frame_for_B))

    # print(training_frame_for_A, training_frame_for_B, testing_frame)

    # Reset column names
    data = data[["0", "1", "2", "3"]]
    data.columns = ['morph_name', 'frame', 'distance_to_A', 'distance_to_B']

    # Find rows for training and testing
    training_data_A = data.loc[data['frame'].isin(training_frame_for_A)]
    training_data_B = data.loc[data['frame'].isin(training_frame_for_B)]
    testing_data = data.loc[data['frame'].isin(testing_frame)]

    # print(training_data_A.shape[0], training_data_B.shape[0], testing_data.shape[0], data.shape[0])
    assert training_data_A.shape[0] + training_data_B.shape[0] + testing_data.shape[0] == data.shape[0]

    """
    Sampling class A:
        Calculate number of samples
        Shuffle the data and take sample
        Note: Always use the same distribution
    """
    if sampling_method != "long_tail":
        nb_training_sample = int(training_data_A.shape[0] * sampling_ratio)
    else:
        nb_training_sample = int(training_data_B.shape[0] * sampling_ratio)

    sample_A = training_data_A.sample(nb_training_sample)

    # For class B, it differs for each sampling method
    if sampling_method == "uniform":
        sample_B = training_data_B.sample(nb_training_sample)

    elif sampling_method == "enriched_tail":
        # Find number of samples for the enriched tails.
        nb_sample_head = int(nb_training_sample * tail_weight)
        nb_sample_tail = nb_sample_head
        nb_sample_middle = nb_training_sample - nb_sample_head - nb_sample_tail

        # Sort all the samples by distance to A or B
        training_data_B_sampled = training_data_B.sample(nb_training_sample)
        # training_data_B_sampled = training_data_B_sampled.sort_values('distance_to_A')
        training_data_B_sampled = training_data_B_sampled.sort_values('distance_to_B')

        # Then do the sampling
        head_samples = training_data_B_sampled.head(nb_sample_head)
        tail_samples = training_data_B_sampled.tail(nb_sample_tail)
        middle_samples = training_data_B_sampled[nb_sample_head:-nb_sample_tail].sample(nb_sample_middle)

        # Combine 3 dataframes
        sample_B = pd.concat([head_samples, middle_samples, tail_samples], ignore_index=True)

    elif sampling_method == "long_tail":
        # Find number of samples for the long tail.
        nb_sample_tail = int(nb_training_sample * tail_weight)
        nb_sample_middle = nb_training_sample - nb_sample_tail

        # Sort values
        training_data_B_sampled = training_data_B.sample(nb_training_sample)
        # training_data_B_sampled = training_data_B_sampled.sort_values('distance_to_A')
        training_data_B_sampled = training_data_B_sampled.sort_values('distance_to_B')

        # Find all samples for each part
        tail_samples = training_data_B_sampled.tail(nb_sample_tail)
        middle_samples =training_data_B_sampled[:-nb_sample_tail].sample(nb_sample_middle)

        # Combine 2 dataframes
        sample_B = pd.concat([middle_samples, tail_samples], ignore_index=True)

    else:
        sample_B = None

    # TODO: Testing samples - use all of them? -JH -> Yup

    if sample_B is not None:
        # To make sure all models can use the same set of sampled data, save these samples
        sub_folder = model_name + "_" + sampling_method + "_nb_train_" + str(nb_training) + "_ratio_" + \
                     str(sampling_ratio) + "_tail_weight_" + str(tail_weight)
        target_dir = os.path.join(save_sample_path, sub_folder)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        # Save samples to target directory
        sample_A.to_csv(os.path.join(target_dir, "sample_a.csv"))
        sample_B.to_csv(os.path.join(target_dir, "sample_b.csv"))
        testing_data.to_csv(os.path.join(target_dir, "test_samples.csv"))

    else:
        raise Exception("Empty sample for class B. Sampling methods supported are: uniform, enriched_tail and long_tail.")



if __name__ == "__main__":
    # sampling(csv_path=uniform_csv_path,
    #          frame_index=all_frames_uniform,
    #          nb_training=5,
    #          sampling_ratio=sampling_ratio,
    #          model_name=model_name,
    #          save_sample_path=data_dir,
    #          sampling_method="uniform")

    # sampling(csv_path=enrich_csv_path,
    #          frame_index=all_frames_enriched,
    #          nb_training=5,
    #          sampling_ratio=sampling_ratio,
    #          model_name=model_name,
    #          save_sample_path=data_dir,
    #          sampling_method="enriched_tail")

    sampling(csv_path=long_tail_csv_path,
             frame_index=all_frames_long,
             nb_training=5,
             sampling_ratio=sampling_ratio,
             model_name=model_name,
             save_sample_path=data_dir,
             sampling_method="long_tail")