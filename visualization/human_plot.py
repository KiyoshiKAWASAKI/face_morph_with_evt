import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, exp, pi
from matplotlib.pyplot import figure
import itertools

human_uniform = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
                "human_uniform_nb_train_4_ratio_0.9.csv"
human_enriched = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/" \
                 "human_enriched_tail_nb_train_4_ratio_0.9_tail_weight_0.3.csv"


all_frames = [8, 20, 30, 40, 45, 50, 55, 60,
              65, 70, 75, 80, 85,
              90, 95, 103, 113, 121, 133, 140]


human = [0.07147375079063880, 0.08383233532934130, 0.08802263439170070, 0.11461951373539600,
         0.1407828282828280, 0.1856513530522340, 0.23733081523449800, 0.32639545884579000,
         0.40833594484487600, 0.526365645721503, 0.631214600377596 ,0.7495285983658080,
         0.8049010367577760, 0.8561536040289580, 0.8927444794952680, 0.9238305941845770,
         0.9286841274850110, 0.936197094125079, 0.9360629921259840, 0.9438485804416400]




def generate_per_frame_response(response_csv_path):
    data = pd.read_csv(response_csv_path)

    # data_selected = data

    # # Create a column for only frame index
    # data_selected['frame_index'] = data_selected.test_img.str. \
    #     split('/').str[-1].str.split("_").str[-1].str.split(".").str[0]
    # data_final = data_selected[["frame_index", "worker_response"]]

    # Group by frame index:
    # response: get counts
    grouped = data.groupby(['frame_index', 'worker_response']).worker_response. \
        agg('count').to_frame('count').reset_index()
    # print(grouped.head(10))

    # responses: get total counts
    grouped_for_counts = grouped[['frame_index', 'count']]
    total_counts = grouped_for_counts.groupby(['frame_index']).sum()
    # print(total_counts)

    # responses: select the counts where worker response is 2
    worker_response_selected = grouped[grouped['worker_response'] == 2]
    worker_response_selected.columns = ["frame_index", "worker_response", "person_b_counts"]
    worker_response_selected = worker_response_selected[["frame_index", "worker_response", "person_b_counts"]]

    person_b_counts = worker_response_selected[["frame_index", "person_b_counts"]]
    # print(worker_response_selected)

    # responses: merge two frames
    result_df = pd.merge(total_counts, person_b_counts, on="frame_index")

    # response: convert to probability of choosing person B as the answer
    result_df["prob_b"] = result_df["person_b_counts"] / result_df["count"]

    # Save results to CSV
    # result_df.to_csv(path_or_buf=save_result_path)

    print(result_df)




if __name__ == "__main__":
    generate_per_frame_response(response_csv_path=human_uniform)
    # generate_per_frame_response(response_csv_path=human_enriched)