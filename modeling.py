
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp, pi
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min, weibull_max, genextreme

#####################################################
# Adjustable params
#####################################################
# TODO: change these for different feature extractors
result_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                  "face_morph_data/distance_files/deepface_euclidean.csv"
model_name = "deepface"

sampling_ratio = 0.8
# sampling_method = "uniform"
# sampling_method = "enriched_tail"
sampling_method = "long_tail"

# data_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
#            "sampled_data/deepface_uniform_nb_train_4_ratio_0.8_tail_weight_0.4"

# data_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
#            "sampled_data/deepface_enriched_tail_nb_train_4_ratio_0.8_tail_weight_0.4"

data_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
           "sampled_data/deepface_long_tail_nb_train_4_ratio_0.8_tail_weight_0.4"

tail_ratio = 0.5
tail_type = "right"
# tail_type = "left"

#####################################################
# Fixed params
#####################################################
save_sample_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/sampled_data"
save_fig_base = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/"

nb_training = 4 # Do not change it - match with human data
all_frames = [8, 20, 30, 40, 45, 50, 55, 60,
              65, 70, 75, 80, 85,
              90, 95, 103, 113, 121, 133, 140]

human = [0.07147375079063880, 0.08383233532934130, 0.08802263439170070, 0.11461951373539600,
         0.1407828282828280, 0.1856513530522340, 0.23733081523449800, 0.32639545884579000,
         0.40833594484487600, 0.526365645721503, 0.631214600377596 ,0.7495285983658080,
         0.8049010367577760, 0.8561536040289580, 0.8927444794952680, 0.9238305941845770,
         0.9286841274850110, 0.936197094125079, 0.9360629921259840, 0.9438485804416400]


def sampling(csv_path,
             frame_index,
             nb_training,
             sampling_ratio,
             sampling_method,
             model_name,
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

        # Sort all the samples by distance to A
        training_data_B = training_data_B.sort_values('distance_to_A')
        training_data_B_sampled = training_data_B.sample(nb_training_sample)

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
        training_data_B = training_data_B.sort_values('distance_to_A')
        training_data_B_sampled = training_data_B.sample(nb_training_sample)

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

    stats = [mean_a, sd_a, var_a, mean_b, sd_b, var_b]

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

    return prob_dict, stats




def evt_fit(data,
            distribution,
            tail_choice,
            tail_ratio):
    """

    :param data: Sampled data
    :param distribution: Weibull or reversed Weibull
    :param tail_choice:
        right - fit tails taken from the right side of sorted samples
        left - fit tails from left side. Scores are flipped by doing (max_score - current_score)
    :param tail_ratio: larger than 0.0, smaller or equal to 0.5
    :return:
    """
    # Format data
    sample = data[["distance_to_A"]]
    sample = sample.distance_to_A.values.tolist()

    # Find the tail samples we will use to fit model
    nb_tail = int(len(sample) * tail_ratio)
    sample = sorted(sample)

    if tail_choice == "right":
        tail_samples = sample[-nb_tail:]

    elif tail_choice == "left":
        max = sample[-1]
        tail_samples_right = sample[-nb_tail:]
        tail_samples = [(max - i) for i in tail_samples_right]

    else:
        raise Exception('Tail choice not implemented. Choose from left or right.')

    # Fit EVT models
    if distribution == "weibull":
        shape, loc, scale = weibull_min.fit(tail_samples)
        return shape, loc, scale

    elif distribution == "reversed_weibull":
        shape, loc, scale = weibull_max.fit(tail_samples)
        return shape, loc, scale

    else:
        raise Exception('Distribution not implemented.')




def evt_predict(data,
                shape,
                loc,
                scale,
                distribution):
    """

    :param data:
    :param shape:
    :param loc:
    :param scale:
    :return:
    """
    sample_w_frame = data[["frame", "distance_to_A"]]
    frames = sample_w_frame['frame'].unique()

    # print("Frames in: ", frames)

    prob_dict = {}
    prob_one_frame = []

    for one_frame in frames:
        frames = sample_w_frame.loc[sample_w_frame['frame'] == one_frame]
        dist = frames["distance_to_A"].tolist()

        for one_dist in dist:
            if distribution == "weibull":
                prob = weibull_min.cdf(one_dist, shape, loc, scale)
            elif distribution == "reversed_weibull":
                prob = weibull_max.cdf(one_dist, shape, loc, scale)
            else:
                raise Exception('Distribution not implemented.')

            prob_one_frame.append(prob)

        prob_dict[one_frame] = mean(prob_one_frame)

    return prob_dict




def evt_model(data_dir,
              distribution,
              tail_type,
              tail_ratio):
    """

    :return:
    """

    # Load training data for class A and B, and testing data
    training_a = pd.read_csv(os.path.join(data_dir, "sample_a.csv"))
    training_b = pd.read_csv(os.path.join(data_dir, "sample_b.csv"))
    testing = pd.read_csv(os.path.join(data_dir, "test_samples.csv"))

    """
    Fit a distribution, currently choose between "weibull" and "reversed_weibull".
    
    weibull: weibull_min. When data is bounded from below.
    reversed weibull:weibull_max. When data is bounded from above.
    
    fit func returns shape, location and scale parameters
    """
    shape_a, loc_a, scale_a = evt_fit(data=training_a,
                                      distribution=distribution,
                                      tail_choice=tail_type,
                                      tail_ratio=tail_ratio)

    shape_b, loc_b, scale_b = evt_fit(data=training_b,
                                      distribution=distribution,
                                      tail_choice=tail_type,
                                      tail_ratio=tail_ratio)

    """
    Use the parameters for our testing samples.
    Probability for each sample is produced by CDF func.
    """
    prob_dict_a = evt_predict(data=training_a,
                                 shape=shape_b,
                                 loc=loc_b,
                                 scale=scale_b,
                                 distribution="weibull")

    prob_dict_b = evt_predict(data=training_b,
                                 shape=shape_b,
                                 loc=loc_b,
                                 scale=scale_b,
                                 distribution="weibull")


    prob_dict_test = evt_predict(data=testing,
                            shape=shape_b,
                            loc=loc_b,
                            scale=scale_b,
                            distribution="weibull")

    prob_dict = {**prob_dict_a, **prob_dict_b, **prob_dict_test}
    prob_dict = dict(sorted(prob_dict.items()))

    print(prob_dict)

    return prob_dict




def plot_prob_curve(gaussian_prob,
                    evt_prob,
                    tail_choice,
                    tail_ratio,
                    human_data,
                    save_fig_dir,
                    sampling,
                    model=model_name):
    """

    :param prob:
    :return:
    """
    list_gauss = sorted(gaussian_prob.items())
    x_gauss, y_gauss = zip(*list_gauss)

    list_evt = sorted(evt_prob.items())
    x_evt, y_evt = zip(*list_evt)

    plt.plot(x_gauss, y_gauss, "-b", label="Gauss")
    plt.plot(x_gauss, y_evt, "-r", label="EVT")
    plt.plot(x_gauss, human_data , "-g", label="Human")

    plt.legend(loc="upper left")

    if not os.path.isdir(os.path.join(save_fig_dir, model)):
        os.mkdir(os.path.join(save_fig_dir, model))

    save_fig_path = save_fig_dir + "/" + model + "/" + str(sampling) + "_tail_side_" + \
                    str(tail_choice) + "_tail_ratio_" + str(tail_ratio) + ".png"

    plt.savefig(save_fig_path)




if __name__ == "__main__":
    # Data Sampling
    # sampling(csv_path=result_csv_path,
    #          frame_index=all_frames,
    #          nb_training=nb_training,
    #          sampling_ratio=sampling_ratio,
    #          model_name=model_name,
    #          sampling_method=sampling_method)

    # Gaussian probabilities
    gaussian_prob, stats = gaussian_naive_bayes(data_dir=data_dir)

    # EVT probabilities
    evt_prob = evt_model(data_dir=data_dir,
                         distribution="weibull",
                         tail_ratio=tail_ratio,
                         tail_type=tail_type)

    # Plot Gaussian and EVT in one figure
    plot_prob_curve(gaussian_prob=gaussian_prob,
                    evt_prob=evt_prob,
                    tail_choice=tail_type,
                    tail_ratio=tail_ratio,
                    human_data=human,
                    sampling=sampling_method,
                    save_fig_dir=save_fig_base)