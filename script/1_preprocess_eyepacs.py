import argparse
import csv
import os
import shutil

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train_images", type=str, required=True)
    arg_parser.add_argument("--train_labels", type=str, required=True)
    arg_parser.add_argument("--test_images", type=str, required=True)
    arg_parser.add_argument("--test_labels", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)
    args = arg_parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(os.path.join(args.output_folder, "images"))

    with open(os.path.join(args.output_folder, "labels.csv"), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['image', 'level'])
        for image_folder, label_csv in [(args.train_images, args.train_labels), (args.test_images, args.test_labels)]:
            df = pd.read_csv(label_csv)
            for _, row in tqdm(df.iterrows(), desc=f"Processing '{label_csv}'", total=len(df)):
                csv_writer.writerow([row['image'], row['level']])
                shutil.copyfile(
                    os.path.join(image_folder, f"{row['image']}.jpeg"),
                    os.path.join(args.output_folder, "images", f"{row['image']}.jpeg")
                )
