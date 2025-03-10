import pandas as pd

class_a_uniform_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                       "sampled_data_0227/EVT/sample_a.csv"
class_b_uniform_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                       "sampled_data_0227/EVT/sample_b_uniform_use_for_both_ab.csv"

class_b_enrich_dist_to_a_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                                "sampled_data_0227/EVT/sample_b_enrich_sort_by_distance_to_a.csv"
class_b_enrich_dist_to_b_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                                "sampled_data_0227/EVT/sample_b_enrich_sort_by_distance_to_b.csv"

class_b_long_dist_to_a_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                              "sampled_data_0227/EVT/sample_b_long_sort_by_distance_to_a.csv"
class_b_long_dist_to_b_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                              "sampled_data_0227/EVT/sample_b_long_sort_by_distance_to_b.csv"


paths = [class_a_uniform_path, class_b_uniform_path,
         class_b_enrich_dist_to_a_path, class_b_enrich_dist_to_b_path,
         class_b_long_dist_to_a_path, class_b_long_dist_to_b_path]


def shuffle_data(data_paths):
    for one_path in data_paths:

        df = pd.read_csv(one_path)
        shuffled_df = df.sample(frac=1)

        save_file_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/sampled_data_0227/"

        shuffled_df.to_csv(save_file_path + one_path.split("/")[-1])




def sort_data(data_path,
              save_path):

    df = pd.read_csv(data_path)
    df_a = df[['frame', 'distance_to_A']]
    df_b = df[['frame', 'distance_to_B']]

    df_a = df_a.sort_values(by=['distance_to_A'])
    df_b = df_b.sort_values(by=['distance_to_B'])

    df_a.to_csv(save_path + "sample_b_uniform_sort_by_distance_to_a.csv")
    df_b.to_csv(save_path + "sample_b_uniform_sort_by_distance_to_b.csv")


if __name__ == "__main__":
    save_path = "/Users/jh5442/Desktop/jov_everything/face_morph_v4_5_sets_modeling_with_3_samplings/" \
                "sampled_data_0227/EVT/"

    sort_data(data_path=class_b_uniform_path,
              save_path=save_path)