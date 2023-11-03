import numpy as np
import sys, os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from numpy import dot
from numpy.linalg import norm
import csv
import pandas as pd
from scipy.spatial import distance



# Path to features
# FaceNet (Google) features
# feature_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/facenet_feat"
# save_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/facenet_euclidean.csv"

# DeepFace (Meta) features
# feature_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/deepface_feat"
# save_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/deepface.csv"

# VggFace - ResNet50
# feature_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/vggface_feat_resnet"
# save_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/vggface_resnet_euclidean.csv"

# TODO: VGGFace - VGG16
# feature_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/vggface_feat_vgg16"
# save_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/vggface_vgg16_euclidean.csv"
# size = 512

# TODO: VGGFace - SE-Net
feature_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/vggface_feat_senet"
save_csv_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/distance_files/vggface_senet_euclidean.csv"
size = 2048



frames = [8, 20, 30, 40, 45, 50, 55, 60,
          65, 70, 75, 80, 85,
          90, 95, 103, 113, 121, 133, 140]
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
                     morph_names,
                     save_result_path,
                     feature_ind,
                     person_a_ind,
                     person_b_ind,
                     nb_rows=6000):
    """

    :param feature:
    :param save_result_path:
    :return:
    """
    all_records = []

    # For each morph, precess each frame
    for one_morph in morph_names:
        # Get person A and person B for this morph
        person_A_feat_path = feature_dir + "/" + one_morph + "_morph_img_" + str(person_a_ind).zfill(4) + ".jpg.npy"
        person_A_feat = np.load(person_A_feat_path, allow_pickle=True)

        person_B_feat_path = feature_dir + "/" + one_morph + "_morph_img_" + str(person_b_ind).zfill(4) + ".jpg.npy"
        person_B_feat = np.load(person_B_feat_path, allow_pickle=True)

        for one_frame in feature_ind:
            # For a frame, find its feature
            feature_name = one_morph + "_morph_img_" + str(one_frame).zfill(4) + ".jpg.npy"
            feature_path = os.path.join(feature_dir, feature_name)

            # Load feature
            feature = np.load(feature_path, allow_pickle=True)

            # Computer distance to person A and person B
            # dist_A = cosine_distance(person_A_feat, feature)
            # dist_B = cosine_distance(person_B_feat, feature)

            dist_A = euclidean_distance(person_A_feat, feature)
            dist_B = euclidean_distance(person_B_feat, feature)

            # print(dist_A, dist_B)

            all_records.append([one_morph, one_frame, dist_A, dist_B])

    result_np = np.asarray(all_records)

    assert result_np.shape[0] == nb_rows
    # np.savetxt(save_result_path, result_np, delimiter=",", fmt="%s")

    # with open(save_result_path, 'w') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(all_records)

    df = pd.DataFrame(all_records)
    df.to_csv(save_result_path)

    print("Result file saved: ", save_result_path)





if __name__ == "__main__":
    # check_features(feature_dir=feature_dir,
    #                shape=size)

    all_morphs = process_feature_files(feature_dir=feature_dir)

    gen_distance_csv(feature_dir=feature_dir,
                     morph_names=all_morphs,
                     save_result_path=save_csv_path,
                     feature_ind=frames,
                     person_a_ind=person_A_ind,
                     person_b_ind=person_B_ind)