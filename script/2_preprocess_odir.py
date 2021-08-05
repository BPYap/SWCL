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
    arg_parser.add_argument("--dev_images", type=str, required=True)
    arg_parser.add_argument("--dev_labels", type=str, required=True)
    arg_parser.add_argument("--test_images", type=str, required=True)
    arg_parser.add_argument("--test_labels", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)
    args = arg_parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(os.path.join(args.output_folder, "images"))
        os.makedirs(os.path.join(args.output_folder, "train"))
        os.makedirs(os.path.join(args.output_folder, "dev"))
        os.makedirs(os.path.join(args.output_folder, "test"))


    def is_normal(keywords):
        if 'normal fundus' in keywords:
            return True
        else:
            keywords = [keyword.strip() for keyword in keywords.split(',')]
            if len(keywords) == 1 and keywords[0] == 'lens dust':
                return True

        return False


    with open(os.path.join(args.output_folder, "labels.csv"), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['image', 'normal-abnormal'])
        for split, image_folder, label_csv in [("train", args.train_images, args.train_labels),
                                               ("dev", args.dev_images, args.dev_labels),
                                               ("test", args.test_images, args.test_labels)]:
            with open(os.path.join(args.output_folder, f"{split}.csv"), 'w', newline='') as w:
                split_csv_writer = csv.writer(w)
                split_csv_writer.writerow(['image', 'normal-abnormal'])

                df = pd.read_excel(label_csv)
                for _, row in tqdm(df.iterrows(), desc=f"Processing '{label_csv}'", total=len(df)):
                    left_image = row['Left-Fundus']
                    left_keywords = row['Left-Diagnostic Keywords']
                    right_image = row['Right-Fundus']
                    right_keywords = row['Right-Diagnostic Keywords']

                    if any(['anterior segment image' in k for k in [left_keywords, right_keywords]]):
                        continue  # skip non-fundus images

                    left_label = 1 if not is_normal(left_keywords) else 0
                    right_label = 1 if not is_normal(right_keywords) else 0
                    csv_writer.writerow([left_image, left_label])
                    csv_writer.writerow([right_image, right_label])
                    shutil.copyfile(
                        os.path.join(image_folder, left_image),
                        os.path.join(args.output_folder, "images", left_image)
                    )
                    shutil.copyfile(
                        os.path.join(image_folder, right_image),
                        os.path.join(args.output_folder, "images", right_image)
                    )

                    split_csv_writer.writerow([left_image, left_label])
                    split_csv_writer.writerow([right_image, right_label])
                    shutil.copyfile(
                        os.path.join(image_folder, left_image),
                        os.path.join(args.output_folder, f"{split}", left_image)
                    )
                    shutil.copyfile(
                        os.path.join(image_folder, right_image),
                        os.path.join(args.output_folder, f"{split}", right_image)
                    )
