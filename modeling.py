
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
# Adjustable params
#####################################################
# TODO: VGG-Face ResNet
model_name = "vggface_resnet"
uniform_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist/vgg_resnet_enriched_tail.csv"
enrich_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist/vgg_resnet_uniform.csv"

# TODO: VGG-Face VGG-16
# model_name = "vggface_vgg16"
# uniform_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist/vgg_vgg16_uniform.csv"
# enrich_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist/vgg_vgg16_enriched_tail.csv"

# TODO: VGG-Face SE-Net
# model_name = "vggface_senet"
# uniform_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist/vgg_senet_uniform.csv"
# enrich_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist/vgg_senet_enriched_tail.csv"
#

sampling_ratio = 0.9


uniform_data_dir = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_modeling/" \
                   "vggface_resnet_uniform_nb_train_5_ratio_0.9_tail_weight_0.4"

enriched_tail_data_dir = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_modeling/" \
                         "vggface_resnet_enriched_tail_nb_train_5_ratio_0.9_tail_weight_0.4"

long_tail_data_dir = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_modeling/" \
                     "vggface_resnet_long_tail_nb_train_5_ratio_0.9_tail_weight_0.4"


# tail_type = "right"
# tail_type = "left"

#####################################################
# Fixed params
#####################################################
save_sample_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_modeling"
save_fig_base = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_fig"

nb_training = 5 # Do not change this - match with human data

all_frames_uniform = [1, 2, 4, 5, 6,
                      36, 38, 40, 42, 45,
                      47, 50, 52, 55, 57,
                      60, 62, 65, 67, 70,
                      72, 74, 76, 78, 80,
                      84, 86, 87, 100, 142]
all_frames_enriched = [1, 2, 4, 5, 6,
                      36, 38, 40, 42, 45,
                      47, 50, 52, 55, 57,
                      60, 62, 65, 67, 70,
                      72, 74, 76, 78, 80,
                      86, 101, 111, 121, 132]

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



def plot_original_samples(csv_path,
                          frame_index,
                          nb_training=4):
    """
    Plot the distribution of original distance data from a model.

    :param csv_path:
    :param frame_index:
    :return:
    """
    # Load distance CSV into data frame
    data = pd.read_csv(csv_path, delimiter=',')

    # Find training frames for A and B, and frames in the gap
    training_frame_for_A = frame_index[:nb_training]
    training_frame_for_B = frame_index[-nb_training:]
    testing_frame = list(set(frame_index) - set(training_frame_for_A) - set(training_frame_for_B))

    # Reset column names
    data = data[["0", "1", "2", "3"]]
    data.columns = ['morph_name', 'frame', 'distance_to_A', 'distance_to_B']

    # Find rows for training and testing
    training_data_A = data.loc[data['frame'].isin(training_frame_for_A)]
    training_data_B = data.loc[data['frame'].isin(training_frame_for_B)]
    testing_data = data.loc[data['frame'].isin(testing_frame)]

    # Get some stats
    print(training_data_A['distance_to_A'].describe())
    print(training_data_B['distance_to_A'].describe())
    print(testing_data['distance_to_A'].describe())

    # Plot all the data
    plt.plot(training_data_A['distance_to_A'], [0]*1200 , "-r", label="training_A")
    plt.plot(training_data_B['distance_to_A'], [2]*1200, "-g", label="training_B")
    plt.plot(testing_data['distance_to_A'], [1]*3600, "-b", label="test")
    plt.legend(loc="upper left")

    save_fig_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" + model_name + "/all_data_plot.png"

    if not os.path.isdir("/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" + model_name):
        os.mkdir("/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" + model_name)

    plt.savefig(save_fig_path)

    plt.clf()

    # Plot histograms to see the data distribution
    # figure(figsize=(10, 30))
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 9), sharey=True, sharex=True)

    ax[0].hist(training_data_A['distance_to_A'], bins=20, label="training_A", width=0.8)
    ax[0].set_title('Training A')
    ax[1].hist(training_data_B['distance_to_A'], bins=20,label="training_B", width=0.8)
    ax[1].set_title('Training B')
    ax[2].hist(testing_data['distance_to_A'], bins=20, label="test", width=0.8)
    ax[2].set_title('Testing')

    fig.tight_layout()

    save_hist_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" + model_name + "/all_data_hist.png"
    plt.savefig(save_hist_path)
    plt.clf()




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
    testing_frame = list(set(frame_index) - set(training_frame_for_A) - set(training_frame_for_B))

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
    # Ignore these for now - JH
    
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

    # print(prob_dict)

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
        tail_samples = sample[:nb_tail]

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
              tail_ratio,
              params=None,
              fit_curve=False,):
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
    if fit_curve:
        shape_a, loc_a, scale_a = evt_fit(data=training_a,
                                          distribution=distribution,
                                          tail_choice=tail_type,
                                          tail_ratio=tail_ratio)

        shape_b, loc_b, scale_b = evt_fit(data=training_b,
                                          distribution=distribution,
                                          tail_choice=tail_type,
                                          tail_ratio=tail_ratio)

    else:
        """
        Use the parameters for our testing samples.
        Probability for each sample is produced by CDF func.
        """
        shape_b = params[0]
        scale_b = params[1]
        loc_b = params[2]

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

    # print(prob_dict)
    # print(shape_b, scale_b, loc_b)

    return prob_dict




def plot_prob_curve(save_fig_dir,
                    params,
                    model,
                    gaussian_prob_uniform,
                    gaussian_prob_enrich,
                    evt_prob_uniform,
                    evt_prob_enrich,
                    human_data_uniform,
                    human_data_enrich,):
    """

    :param prob:
    :return:
    """
    # Gaussian probabilities for uniform and enriched tail
    list_gauss_uniform = sorted(gaussian_prob_uniform.items())
    x_gauss_uniform, y_gauss_uniform = zip(*list_gauss_uniform)

    list_gauss_enrich = sorted(gaussian_prob_enrich.items())
    x_gauss_enrich, y_gauss_enrich = zip(*list_gauss_enrich)

    # EVT probabilities for uniform and enriched tail
    list_evt_uniform = sorted(evt_prob_uniform.items())
    x_evt_uniform, y_evt_uniform = zip(*list_evt_uniform)

    list_evt_enrich = sorted(evt_prob_enrich.items())
    x_evt_enrich, y_evt_enrich = zip(*list_evt_enrich)

    # Plot 6 curves
    plt.plot(x_gauss_uniform, y_gauss_uniform, label="Gaussain Uniform")
    plt.plot(x_gauss_uniform, y_gauss_enrich, label="Gaussain Enriched Tail")
    plt.plot(x_gauss_uniform, y_evt_uniform, label="EVT Uniform")
    plt.plot(x_gauss_uniform, y_evt_enrich, label="EVT Enriched Tail")
    plt.plot(human_test_ind, human_data_uniform , label="Human Uniform")
    plt.plot(human_test_ind, human_data_enrich , label="Human Enriched Tail")

    plt.legend(loc="lower right")

    if not os.path.isdir(os.path.join(save_fig_dir, model)):
        os.mkdir(os.path.join(save_fig_dir, model))

    save_fig_path = save_fig_dir + "/" + model + "/tail_ratio_" + str(params[-1]) + \
                    "_shape_" + str(params[0]) + \
                    "_scale_" + str(params[1]) + "_loc_" + str(params[2]) + ".png"

    plt.savefig(save_fig_path)
    plt.clf()




if __name__ == "__main__":
    # TODO: Data Sampling - only need to run this again
    #  when sampling parameters are changed
    # sampling(csv_path=uniform_csv_path,
    #          frame_index=all_frames_uniform,
    #          nb_training=nb_training,
    #          sampling_ratio=sampling_ratio,
    #          model_name=model_name,
    #          sampling_method="uniform")
    #
    # sampling(csv_path=enrich_csv_path,
    #          frame_index=all_frames_enriched,
    #          nb_training=nb_training,
    #          sampling_ratio=sampling_ratio,
    #          model_name=model_name,
    #          sampling_method="enriched_tail")
    #
    # sampling(csv_path=enrich_csv_path,
    #          frame_index=all_frames_enriched,
    #          nb_training=nb_training,
    #          sampling_ratio=sampling_ratio,
    #          model_name=model_name,
    #          sampling_method="long_tail")

    # TODO: Gaussian probabilities
    gaussian_prob_uniform, stats_uniform = gaussian_naive_bayes(data_dir=uniform_data_dir)
    print("Uniform Gaussian prob from sklearn:")
    print(gaussian_prob_uniform)

    gaussian_prob_enrich, stats_enrich = gaussian_naive_bayes(data_dir=enriched_tail_data_dir)
    print("Enriched tail Gaussian prob from sklearn:")
    print(gaussian_prob_enrich)

    # TODO: EVT probabilities
    """
    Grid search for best parameter combos.
    First round is a rough search. Use the same parameters for all setups
    """
    # First round
    # shape = np.arange(0.1, 0.6, 0.1).tolist()
    # scale = np.arange(1.0, 2.0, 0.1).tolist()
    # location = np.arange(50, 70, 2).tolist()

    # Second round
    shape = np.arange(0.1, 2.0, 0.1).tolist()
    scale = np.arange(0.1, 2.0, 0.1).tolist()
    location = np.arange(2, 100, 2).tolist()

    # print(len(shape), len(scale), len(location))
    tail_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

    shape = [round(item, 3) for item in shape]
    scale = [round(item, 3) for item in scale]
    location = [round(item, 1) for item in location]

    all_params = list(itertools.product(*[shape, scale, location, tail_ratio]))
    print(len(all_params))
    # print(all_params[0])
    # sys.exit()

    for one_param in all_params:
        # EVT Uniform
        evt_prob_uniform = evt_model(data_dir=uniform_data_dir,
                                     distribution="weibull",
                                     tail_ratio=one_param[-1],
                                     tail_type=None,
                                     params=one_param,
                                     fit_curve=False)

        # EVT Enriched Tail
        evt_prob_enrich = evt_model(data_dir=enriched_tail_data_dir,
                                     distribution="weibull",
                                     tail_ratio=one_param[-1],
                                     tail_type=None,
                                     params=one_param,
                                     fit_curve=False)

        # Plot Gaussian and EVT in one figure
        plot_prob_curve(gaussian_prob_uniform=gaussian_prob_uniform,
                        gaussian_prob_enrich=gaussian_prob_enrich,
                        evt_prob_uniform=evt_prob_uniform,
                        evt_prob_enrich=evt_prob_enrich,
                        human_data_uniform=human_uniform_prob,
                        human_data_enrich=human_enrich_prob,
                        save_fig_dir=save_fig_base,
                        params=one_param,
                        model=model_name)

        # sys.exit()