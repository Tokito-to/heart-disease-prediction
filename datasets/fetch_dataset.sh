#!/usr/bin/bash

curl -Lo heart-disease-dataset.zip https://www.kaggle.com/api/v1/datasets/download/abdmental01/heart-disease-dataset
curl -Lo heart-dataset.zip https://www.kaggle.com/api/v1/datasets/download/mfarhaannazirkhan/heart-dataset
curl -Lo Cardiovascular_Disease_Dataset.csv https://data.mendeley.com/public-files/datasets/dzz48mvjht/files/2ee83017-9af4-4618-b340-b8594accc63f/file_downloaded
find . -name "*.zip" -exec unzip -o {} \;
find . -type f -not \( -name "*.csv" -o -name "*.sh" -o -name "*.py" \) -delete
