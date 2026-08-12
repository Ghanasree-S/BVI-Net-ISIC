#!/usr/bin/env bash
# Downloads ISIC2018 Task 1 (Lesion Boundary Segmentation) training images + ground truth.
# Source: https://challenge.isic-archive.com/data/#2018
set -e

DEST="$(dirname "$0")/isic2018_raw"
mkdir -p "$DEST"
cd "$DEST"

IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"
GT_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"

echo "Downloading training images..."
curl -L -o images.zip "$IMG_URL"
echo "Downloading ground truth masks..."
curl -L -o masks.zip "$GT_URL"

echo "Extracting..."
unzip -q -o images.zip
unzip -q -o masks.zip

echo "Done. Raw data in: $DEST"
echo "Next: python data/prepare_isic_a.py"
