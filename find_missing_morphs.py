"""
4 morphs sequences are missing.
There should be 300 in total but there are 296 in img_json.
Find the missing morphs and generate the sequences again.

"""
import json
import os, sys
import numpy as np
import shutil



# These 2 adding up together has 300 morphs
survey_294_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/survey_lists"
# survey_294_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_300_updated/52_survey_lists/"
survey_plus_6_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/survey_lists_293_plus"

# This dir contains 296 morphs
original_morph_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/normal_quests"
processed_morph_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/normal_quests_processed"
control_file = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/control_list.npy"


# source and target for copying 241 morphs
# source = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/face_morph_293_plus"
# target = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_300_updated/241_morphs"

# source_processed = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/normal_quests_processed"
# target_processed = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_300_updated/241_morphs_processed"
#
# source_feat = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/deepface_feat"
# target_feat = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_300_updated/241_morph_features/deepface_feat"

correct_name = "Yana_Klochkova_to_Thomas_Cloyd"
survey_file = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/face_morph_survey_169.npy"



def find_missing_morphs(morph_dir,
                        processed_morph_dir,
                        survey_dir,
                        control_path,
                        nb_control=5,
                        nb_questions=25):
    """

    :param survey_dir:
    :param img_json_path:
    :return:
    """
    # Check original morphs and processed morphs
    original_morphs = os.listdir(morph_dir)
    processed_morphs = os.listdir(processed_morph_dir)

    if original_morphs.sort() == processed_morphs.sort():
        print("True")
    else:
        print("False")

    # First, find all the controls
    control = np.load(control_path, allow_pickle=True)

    control_morphs = []
    for i in range(nb_control):
        control_morphs.append(control.item()[i]["class_a_images"][0].split("/")[-2])
    print(control_morphs)

    # Then find 300 morphs.
    morph_names = []
    for one_dir in survey_dir:
        all_surveys = os.listdir(one_dir)

        for one_survey in all_surveys:
            one_survey_path = os.path.join(one_dir, one_survey)
            one_survey = np.load(one_survey_path, allow_pickle=True)

            for j in range(nb_questions):
                one_morph_name = one_survey.item()[j]["class_a_images"][0].split("/")[-2]

                if (one_morph_name not in morph_names) and (one_morph_name not in control_morphs):
                    # print(j, one_morph_name, one_survey_path)
                    morph_names.append(one_morph_name)
                    break

    print(len(morph_names))

    # Last, find missing morphs
    missing = list(set(morph_names) - set(processed_morphs))
    print(missing)


    """
    'Phoenix_Chang_to_Tatiana_Shchegoleva', (generate)
    'Yana_Klochkova_to_Thomas_Cloyd', => just needs to be processed
    'Alexander_Payne_to_Jeffrey_Pfeffer', (generate)
    'Saoud_Al_Faisal_to_Martha_Beatriz_Roque' (generate)
    """





def fix_wrong_morph(npy_file_path,
                    morph_name,
                    control_file_path,
                    nb_control=5):
    """

    :param npy_file_path:
    :param morph_name:
    :param control_file_path:
    :param nb_control:
    :return:
    """
    # First, find all the controls
    control = np.load(control_file_path, allow_pickle=True)

    control_morphs = []
    for i in range(nb_control):
        control_morphs.append(control.item()[i]["class_a_images"][0].split("/")[-2])
    print(control_morphs)

    # Load the morph that has wrong morph name
    wrong_survey = np.load(npy_file_path, allow_pickle=True)
    print(wrong_survey)





def copy_morphs(morphs,
                source,
                target):
    """
    There are 271 morphs that have both image and data.
    Copy those to a new location to be clear.

    :param morphs:
    :param source:
    :param target:
    :return:
    """

    for one_morph in morphs:
        if os.path.isdir(os.path.join(source, one_morph)):
            shutil.copytree(os.path.join(source, one_morph), os.path.join(target, one_morph))


    print(len(os.listdir(target)))



def copy_feature(morphs,
                 source,
                 target):
    """
    There are 271 morphs that have both image and data.
    Copy those to a new location to be clear.

    :param morphs:
    :param source:
    :param target:
    :return:
    """

    for one_morph in morphs:
        if os.path.isdir(os.path.join(source, one_morph)):
            shutil.copytree(os.path.join(source, one_morph), os.path.join(target, one_morph))


    print(len(os.listdir(target)))




def mv_morph(morphs,
            source,
            target):
    """
    There are 271 morphs that have both image and data.
    Move those to a new location to be clear.

    :param morphs:
    :param source:
    :param target:
    :return:
    """

    for one_morph in morphs:
        print(os.path.join(source, one_morph))
        if os.path.isdir(os.path.join(source, one_morph)):
            shutil.move(os.path.join(source, one_morph), os.path.join(target, one_morph))
        else:
            print("no such dir")


    print(len(os.listdir(target)))



def check_all_morphs(morph_dir):
    """

    :param morph_dir:
    :return:
    """
    all_morphs = os.listdir(morph_dir)


    for one_morph in all_morphs:
        print(len(os.listdir(os.path.join(morph_dir, one_morph))))

    print(len(all_morphs))



if __name__ == "__main__":
    morph_380 = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_380/morphs"
    check_all_morphs(morph_dir=morph_380)
    # find_missing_morphs(morph_dir=original_morph_dir,
    #                     processed_morph_dir=processed_morph_dir,
    #                     survey_dir=[survey_294_dir, survey_plus_6_dir],
    #                     control_path=control_file)

    #
    # fix_wrong_morph(npy_file_path=survey_file,
    #                 morph_name=correct_name,
    #                 control_file_path=control_file,
    #                 nb_control=5)


    # print(len(morph_248))

    """
    236 in: normal_quests
    5 in: face_morph_293_plus
    Total: 241
    """
    # copy_morphs(morphs=morph_248,
    #             source=source,
    #             target=target)

    # copy_morphs(morphs=morph_248,
    #             source=source_processed,
    #             target=target_processed)

    # TODO: copy features
    # copy_feature(morphs=morph_248,
    #              source=source_feat,
    #              target=target_feat)

    # TODO: move used morphs
    # # move_morph_source = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/normal_quests" #236
    # move_morph_source = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/face_morph_293_plus" #5
    # move_morph_target = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/used_241"
    #
    # mv_morph(morphs=morph_248,
    #          source=move_morph_source,
    #          target=move_morph_target)