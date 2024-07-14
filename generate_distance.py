import numpy as np
import sys, os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from numpy import dot
from numpy.linalg import norm
import csv
import pandas as pd
from scipy.spatial import distance



# Paths
feature_dir = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_features/"
save_csv_path = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_dist"


person_A_ind = 1
person_B_ind = 142




def check_features(feature_dir,
                   shape):
    """
    Check all the features and remove the ones that have wrong size

    :param feature_dir:
    :param shape:
    :return:
    """
    all_features = os.listdir(feature_dir)

    for one_feature in all_features:
        feature_path = os.path.join(feature_dir, one_feature)

        feature = np.load(feature_path, allow_pickle=True)

        if feature.shape[0] != shape:
            os.remove(feature_path)
            print("Incorrect shape: ", feature.shape)




def cosine_distance(feature_a,
                    feature_b):
    """

    :param feature_a:
    :param feature_b:
    :return:
    """
    return 1 - dot(feature_a, feature_b)/(norm(feature_a)*norm(feature_b))




def euclidean_distance(feature_a,
                       feature_b):

    return distance.euclidean(feature_a, feature_b)




def process_feature_files(feature_dir,
                          nb_total=300):
    """

    :param feature_dir:
    :return:
    """
    all_features = os.listdir(feature_dir)
    all_features.sort()

    # Find all the morphs
    all_morphs = []

    for one_feature in all_features:
        one_morph = one_feature.split(".")[0].split("_")[:-3]
        one_morph = '_'.join(one_morph)

        if one_morph not in all_morphs:
            all_morphs.append(one_morph)

    assert len(all_morphs) == nb_total

    return all_morphs




def gen_distance_csv(feature_dir,
                     save_result_path,
                     person_a_ind,
                     person_b_ind):
    """

    :param feature:
    :param save_result_path:
    :return:
    """
    all_models = os.listdir(feature_dir)

    """
    print("All models: ", all_models)
    
    All models:  ['deepface', 'facenet', 'vgg_resnet', 'vgg_senet', 'vgg_vgg16']
    """

    # For each morph, precess each frame
    for one_model in all_models:
        all_config = os.listdir(os.path.join(feature_dir, one_model))
        # print(all_config) # ['enriched_tail', 'uniform']

        for one_config in all_config:
            print("Dir: ", os.path.join(feature_dir, one_model, one_config))

            all_feature = os.listdir(os.path.join(feature_dir, one_model, one_config))

            all_records = []
            for one_feature in all_feature:
                """
                # print(one_feature) # Aaron_Guiel_to_Andy_Graves_morph_img_0001.jpg.npy
                # print(morph_name) # Aaron_Guiel_to_Andy_Graves
                """
                # Find name of this morph
                morph_name = "_". join(one_feature.split(".")[0].split("_")[:-3])
                frame = int(one_feature.split(".")[0].split("_")[-1])
                feature = np.load(os.path.join(feature_dir, one_model, one_config, one_feature), allow_pickle=True)

                # Feature for A and B
                person_A_feat_path = os.path.join(feature_dir, one_model) + \
                                     "/uniform/" + morph_name + "_morph_img_0001.jpg.npy"
                person_B_feat_path = os.path.join(feature_dir, one_model) + \
                                     "/enriched_tail/" + morph_name + "_morph_img_0142.jpg.npy"

                person_A_feat = np.load(person_A_feat_path, allow_pickle=True)
                person_B_feat = np.load(person_B_feat_path, allow_pickle=True)

                dist_A = euclidean_distance(person_A_feat, feature)
                dist_B = euclidean_distance(person_B_feat, feature)

                all_records.append([morph_name, frame, dist_A, dist_B])

            result_np = np.asarray(all_records)
            print("Number of records: ", result_np.shape[0])

            result_file_path = save_result_path + "/" + one_model + "_" + one_config + ".csv"

            df = pd.DataFrame(all_records)
            df.to_csv(result_file_path)

            print("Result file saved: ", result_file_path)




if __name__ == "__main__":
    gen_distance_csv(feature_dir=feature_dir,
                     save_result_path=save_csv_path,
                     person_a_ind=person_A_ind,
                     person_b_ind=person_B_ind)