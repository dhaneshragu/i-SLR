import os
import csv
import pandas as pd
from tqdm import tqdm

base_directory = "INCLUDE"  # Name of the directory where files are there
"""
Assumed this structure:
-> INCLUDE
    -> Adjectives
        -> Happy
        -> Sad
        .
        .
    -> People
        -> Father
        -> Mother
        .
        .
    .
"""
csv_file_path = 'train-preprocessed.csv'


with open(csv_file_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for sign_directory in tqdm(os.listdir(base_directory),desc='processing INCLUDE',unit='dir'):
        sign_directory_path = os.path.join(base_directory, sign_directory) ## INCLUDE/Adjectives/
        if os.path.isdir(sign_directory_path):
            for sign_name in tqdm(os.listdir(sign_directory_path),desc='processing different signs',unit='sign'):
                video_file_path = os.path.join(sign_directory_path, sign_name) ## INCLUDE/Adjectives/Happy/
                
                for video_names in tqdm(os.listdir(video_file_path),desc='processing different videos of a sign',unit='videos'):
                    path_ = os.path.join(video_file_path, video_names) 
                    print("p", path_)
                    command = f'python preprocess.py --input "{path_}"'
                    os.system(command)

                    directory, filename_ext = os.path.split(path_)
                    filename, extension = os.path.splitext(filename_ext)

                    # Replace the directory name
                    new_directory = directory.replace("INCLUDE", "islr-xyz")

                    # Create the new file path by joining the new directory, filename, and .csv extension
                    csv_name = os.path.join(new_directory, filename + ".csv")

                    sign_name_csv = sign_name.split('.')[-1].strip()
                    label = int(sign_name.split('.')[-2])
                    csv_writer.writerow([csv_name, sign_name_csv, sign_directory])

# df = pd.read_csv("train-preprocessed.csv")
# print(df.head())
