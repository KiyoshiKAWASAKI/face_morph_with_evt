
import sys, os
import numpy as np
import matplotlib as plt
import pandas as pd
from math import sqrt, exp, pi
from sklearn.naive_bayes import GaussianNB


# Paths for result files
result_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/deepface.csv"
model_name = "deepface"

save_sample_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/sampled_data"

data_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
           "sampled_data/uniform_nb_train_3_ratio_0.8_tail_weight_0.4"


# Set number of training frames as a parameter
nb_training = 3
sampling_ratio = 0.8
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
        # To make sure all models can use the same set of sampled data, save these samples
        sub_folder = sampling_method + "_nb_train_" + str(nb_training) + "_ratio_" + \
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




# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))




# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)




# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    # print((1 / (sqrt(2 * pi) * stdev)) * exponent)
    return (1 / (sqrt(2 * pi) * stdev)) * exponent




def get_prob(data_frame,
             classifier):
    """

    :param data_frame:
    :return:
    """
    sample_w_frame = data_frame[["frame", "distance_to_A"]]
    frames = sample_w_frame['frame'].unique()

    print("Frames in: ", frames)

    prob_dict = {}
    prob_one_frame = []

    for one_frame in frames:
        frames = sample_w_frame.loc[sample_w_frame['frame'] == one_frame]
        dist = frames["distance_to_A"].tolist()

        for one_dist in dist:
            prob = classifier.predict_proba(np.asarray(one_dist).reshape(-1, 1))
            prob_one_frame.append(prob[0][1])

        prob_dict[one_frame] = mean(prob_one_frame)

    return prob_dict




def gaussian_naive_bayes(data_dir):
    """
    Use class A and class B training data to fit a gaussian model,
    then use bayesian theorem to calculate probability.

    :return:
    """
    # Step 1: Load training data for class A and B, and testing data
    training_a = pd.read_csv(os.path.join(data_dir, "sample_a.csv"))
    training_b = pd.read_csv(os.path.join(data_dir, "sample_b.csv"))
    testing = pd.read_csv(os.path.join(data_dir, "test_samples.csv"))

    # Take feature column for training data and convert data to list
    a_sample = training_a[["distance_to_A"]]
    b_sample = training_b[["distance_to_A"]]

    a_sample = a_sample.distance_to_A.values.tolist()
    b_sample = b_sample.distance_to_A.values.tolist()

    print("Number of samples in class A:", len(a_sample))
    print("Number of samples in class B:", len(b_sample))

    # Step 2: Calculate mean and standard deviation
    mean_a = mean(a_sample)
    sd_a = stdev(a_sample)
    var_a = np.var(np.asarray(a_sample))

    mean_b = mean(b_sample)
    sd_b = stdev(b_sample)
    var_b = np.var(np.asarray(b_sample))

    print("Mean for A: ", mean_a)
    print("SD for A: ", sd_a)
    print("Var for A: ", var_a )
    print("Mean for B: ", mean_b)
    print("SD for B :", sd_b)
    print("Var for B: ", var_b)

    """
    # Ignore these for now 0 JH
    
    # Step 3: Gaussian Probability Density Function.
        # -- defined above

    # Step 4: Class Probabilities -- P(class|data) = P(X|class) * P(class)
    # Calculate avg prob of choosing B for each frame, respectively
    """

    # Build and fit Gaussian Naive Bayes Model
    clf = GaussianNB()

    X = np.asarray(a_sample + b_sample)
    Y = np.asarray([0] * len(a_sample) + [1] * len(b_sample))
    clf.fit(X.reshape(-1, 1), Y)

    # Training data A
    prob_dict_a = get_prob(data_frame=training_a,
                            classifier=clf)

    # Training data B
    prob_dict_b = get_prob(data_frame=training_b,
                           classifier=clf)

    # Testing data
    prob_dict_test = get_prob(data_frame=testing,
                         classifier=clf)

    # Combine dictionaries and sort
    prob_dict = {**prob_dict_a, **prob_dict_b, **prob_dict_test}
    prob_dict = dict(sorted(prob_dict.items()))
    print(prob_dict)




def plot_prob_curve(prob_dict):
    """

    :param prob:
    :return:
    """
    pass




def evt_naive_bayes(training_data,
                    testing_data,
                    distribution):
    """

    :param training_data:
    :param testing_data:
    :param distribution:
    :return:
    """

    pass





if __name__ == "__main__":
    # sampling(csv_path=result_csv_path,
    #          frame_index=all_frames,
    #          nb_training=nb_training,
    #          sampling_ratio=sampling_ratio,
    #          sampling_method=sampling_method)

    gaussian_naive_bayes(data_dir=data_dir)
