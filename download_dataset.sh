#!/bin/bash
# Make sure you have exported KAGGLE_USERNAME and KAGGLE_KEY
# or have placed your kaggle.json in ~/.kaggle/kaggle.json

echo "Installing kaggle CLI via pip..."
pip install kaggle

echo "Downloading Deep-SAR Oil Spill Segmentation dataset..."
# The exact Kaggle dataset identifier might be 'mikhailhushchyn/oil-spill-detection-dataset'
# Or for the refined version: 'someone/deep-sar-oil-spill-segmentation-refined'
# If this slug is incorrect, please update it with the exact Kaggle slug.
DATASET_SLUG="mikhailhushchyn/oil-spill-detection-dataset" 

kaggle datasets download -d $DATASET_SLUG

echo "Extracting dataset..."
mkdir -p dataset
unzip -q -o *.zip -d dataset/

# You may need to organize the extracted files into dataset/images and dataset/masks 
# if they are not already in that format.
echo "Dataset downloaded and extracted."
