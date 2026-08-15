#!/usr/bin/env bash
# Downloads ISIC2018's OFFICIAL validation set (images + ground truth masks),
# so our validation split can match the paper's exactly (100 images).
# Note: the official TEST set (1000 images) has never had its ground truth
# masks publicly released -- there is no way to exactly reproduce that split.
set -e

DEST="$(dirname "$0")/isic2018_val_raw"
mkdir -p "$DEST"
cd "$DEST"

VAL_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip"
VAL_GT_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip"

echo "Downloading official validation images..."
curl -L -o val_images.zip "$VAL_IMG_URL"
echo "Downloading official validation ground truth..."
curl -L -o val_gt.zip "$VAL_GT_URL"

echo "Extracting..."
unzip -q -o val_images.zip
unzip -q -o val_gt.zip

echo "Done. Official validation data in: $DEST"
