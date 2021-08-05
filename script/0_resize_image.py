import argparse
import os

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--image_folder", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)
    arg_parser.add_argument("--size", type=int, required=True)
    args = arg_parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    count = 1
    listing = os.listdir(args.image_folder)
    for file in listing:
        print(f"\rprocessing images ({count}/{len(listing)})", end='', flush=True)
        try:
            image = Image.open(os.path.join(args.image_folder, file))
            new_image = transforms.functional.resize(image, args.size)
            new_image.save(os.path.join(args.output_folder, file))
        except UnidentifiedImageError:
            continue
        count += 1
