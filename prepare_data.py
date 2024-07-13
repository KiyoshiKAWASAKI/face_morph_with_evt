import os
import sys
import shutil


all_morph_sequence_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/" \
                          "jhuang24/face_morph_v4_5_sets"
save_data_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                 "face_morph_v4_5_sets_processed"

train_uniform_a_frames = [1, 2, 4, 5, 6]
train_uniform_b_frames = [86, 101, 111, 121, 132]
train_enrich_b_frames = [84, 86, 87, 100, 142]


testing_frames = [36, 38, 40, 42, 45,
                  47, 50, 52, 55, 57,
                  60, 62, 65, 67, 70,
                  72, 74, 76, 78, 80]


def process_morph_data(source_path,
                       target_path,
                       class_a_frames,
                       uniform_b_frames,
                       enrich_b_frames,
                       testing_frames):
    """
    For each morph sequence, only select the frames that have human data.
    And save these into a separate folder.

    :param source_path:
    :param target_path:
    :return:
    """
    # List all morph sequences
    all_morphs = os.listdir(source_path)
    print(len(all_morphs))

    # Process each sequence
    for one_morph in all_morphs:
        one_morph_folder = os.path.join(source_path, one_morph)

        # Check whether a morph exists, if not, make a dir for it
        target_save_folder = os.path.join(target_path, one_morph)
        # if not os.path.isdir(target_save_folder):
        #     print("Making new directory: ", target_save_folder)
        #     os.mkdir(target_save_folder)

        # Make directories for training and testing frames
        target_uniform_folder = os.path.join(target_save_folder, "uniform")
        # os.mkdir(target_uniform_folder)
        target_enrich_folder = os.path.join(target_save_folder, "enriched_tail")
        # os.mkdir(target_enrich_folder)

        # A => uniform and enriched tail
        for one_frame in class_a_frames:
            frame_name = str(one_frame).zfill(4)
            frame_name = "morph_img_" + frame_name + ".jpg"

            src = os.path.join(one_morph_folder, frame_name)
            dst = os.path.join(target_uniform_folder, frame_name)
            # print(src)
            # print(dst)
            shutil.copyfile(src, dst)

        for one_frame in class_a_frames:
            frame_name = str(one_frame).zfill(4)
            frame_name = "morph_img_" + frame_name + ".jpg"

            src = os.path.join(one_morph_folder, frame_name)
            dst = os.path.join(target_enrich_folder, frame_name)
            shutil.copyfile(src, dst)

        # Uniform B => uniform
        for one_frame in uniform_b_frames:
            frame_name = str(one_frame).zfill(4)
            frame_name = "morph_img_" + frame_name + ".jpg"

            src = os.path.join(one_morph_folder, frame_name)
            dst = os.path.join(target_uniform_folder, frame_name)
            shutil.copyfile(src, dst)

        # Enriched B => enriched tail
        for one_frame in enrich_b_frames:
            frame_name = str(one_frame).zfill(4)
            frame_name = "morph_img_" + frame_name + ".jpg"

            src = os.path.join(one_morph_folder, frame_name)
            dst = os.path.join(target_enrich_folder, frame_name)
            shutil.copyfile(src, dst)

        # Test => uniform and enriched tail
        for one_frame in testing_frames:
            frame_name = str(one_frame).zfill(4)
            frame_name = "morph_img_" + frame_name + ".jpg"

            src = os.path.join(one_morph_folder, frame_name)
            dst = os.path.join(target_uniform_folder, frame_name)
            shutil.copyfile(src, dst)

        for one_frame in testing_frames:
            frame_name = str(one_frame).zfill(4)
            frame_name = "morph_img_" + frame_name + ".jpg"

            src = os.path.join(one_morph_folder, frame_name)
            dst = os.path.join(target_enrich_folder, frame_name)
            shutil.copyfile(src, dst)




if __name__ == "__main__":
    all_surveys = os.listdir(all_morph_sequence_path)

    for one_survey in all_surveys:
        process_morph_data(source_path=os.path.join(all_morph_sequence_path, one_survey),
                           target_path=save_data_path,
                           class_a_frames=train_uniform_a_frames,
                           uniform_b_frames=train_uniform_b_frames,
                           enrich_b_frames=train_enrich_b_frames,
                           testing_frames=testing_frames)