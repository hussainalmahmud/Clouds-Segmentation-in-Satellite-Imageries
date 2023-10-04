#!/bin/bash
# run in terminal with command: bash dataset_download.sh or ./dataset_download.sh

# Create a directory called "dataset" if it doesn't exist
mkdir -p dataset

# Move to the "dataset" directory
cd dataset

# Download dataset (Uncomment the dataset you want to download)
# wget "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/cloudmaskcompetition/sample.zip" -O sample.zip
# wget "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/cloudmaskcompetition/train_true_color.zip" -O train_true_color.zip
# wget "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/cloudmaskcompetition/train_mask.zip" -O train_mask.zip
# wget "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/cloudmaskcompetition/evaluation_true_color.zip" -O evaluation_true_color.zip

# Unzip the downloaded files (Uncomment the dataset you want to unzip)
# unzip ../dataset/sample.zip -d ../dataset/sample
# unzip ../dataset/train_true_color.zip -d ../dataset/train_true_color
# unzip ../dataset/train_mask.zip -d ../dataset/train_mask
# unzip ../dataset/evaluation_true_color.zip -d ../dataset/evaluation_true_color

echo "Data downloaded and extracted successfully into the 'dataset' directory!"

# Delete the downloaded zip files (Uncomment the dataset you want to delete)
# rm ../dataset/sample.zip
# rm ../dataset/train_true_color.zip
# rm ../dataset/train_mask.zip
# rm ../dataset/evaluation_true_color.zip

echo "All zip files deleted successfully!"