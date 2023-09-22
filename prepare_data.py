import os
import sys
import shutil


all_morph_sequence_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/4_missing_morphs"
save_data_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/4_missing_morphs_processed"

training_frames = [1, 8, 20, 30, 121, 133, 140, 142]
testing_frames = [40, 45, 50, 55, 60,
                  65, 70, 75, 80, 85,
                  90, 95, 103, 113]


def process_morph_data(source_path,
                       target_path,
                       training_frames,
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
        one_morph_folder = os.path.join(all_morph_sequence_path, one_morph)

        # Check whether a morph exists, if not, make a dir for it
        target_save_folder = os.path.join(target_path, one_morph)
        if not os.path.isdir(target_save_folder):
            print("Making new directory: ", target_save_folder)
            os.mkdir(target_save_folder)

            # Make directories for training and testing frames
            target_training_folder = os.path.join(target_save_folder, "training")
            os.mkdir(target_training_folder)
            target_testing_folder = os.path.join(target_save_folder, "testing")
            os.mkdir(target_testing_folder)

            # Copying and pasting training and testing frames to their folders, respectively
            for one_frame in training_frames:
                frame_name = str(one_frame).zfill(4)
                frame_name = "morph_img_" + frame_name + ".jpg"

                src = os.path.join(one_morph_folder, frame_name)
                dst = os.path.join(target_training_folder, frame_name)
                shutil.copyfile(src, dst)

            for one_frame in testing_frames:
                frame_name = str(one_frame).zfill(4)
                frame_name = "morph_img_" + frame_name + ".jpg"

                src = os.path.join(one_morph_folder, frame_name)
                dst = os.path.join(target_testing_folder, frame_name)
                shutil.copyfile(src, dst)




if __name__ == "__main__":
    process_morph_data(source_path=all_morph_sequence_path,
                       target_path=save_data_path,
                       training_frames=training_frames,
                       testing_frames=testing_frames)