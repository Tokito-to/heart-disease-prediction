#!/usr/bin/bash

curl -Lo heart-disease-dataset.zip https://www.kaggle.com/api/v1/datasets/download/abdmental01/heart-disease-dataset
curl -Lo heart-dataset.zip https://www.kaggle.com/api/v1/datasets/download/mfarhaannazirkhan/heart-dataset
find . -name "*.zip" -exec unzip -o {} \;
find . -type f -not \( -name "*cleaned*" -o -name "*.sh" -o -name "*.py" \) -delete