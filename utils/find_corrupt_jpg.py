import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", help="Path to folder with images to check")
args = parser.parse_args()

for root, dirnames, filenames in os.walk(args.imgdir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_path = os.path.join(root, filename)
            print(img_path)
            img = cv2.imread(img_path)
