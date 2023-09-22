import sys
import os
import json



img_folder_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                  "face_morph_data/4_missing_morphs_processed/"
save_json_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                 "face_morph_data/4_missing_morph_imgs.json"




def generate_json_files(input_dir,
                        save_json_path):
    """

    :param input_dir:
    :param save_json_path:
    :return:
    """

    img_path_dict = {}

    for path, subdirs, files in os.walk(input_dir):
        for one_file_name in files:
            full_path = os.path.join(path, one_file_name)

            one_file_dict = {}
            one_file_dict["img_path"] = full_path

            img_path_dict[len(img_path_dict) + 1] = one_file_dict

    with open(save_json_path, 'w') as f:
        json.dump(img_path_dict, f)
        print("Saving file to %s" % save_json_path)




if __name__ == "__main__":
    generate_json_files(input_dir=img_folder_path,
                        save_json_path=save_json_path)