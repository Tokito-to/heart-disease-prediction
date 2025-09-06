# Heart Disease Predection

## DataSet Source
- https://www.kaggle.com/datasets/abdmental01/heart-disease-dataset
- https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset
- https://data.mendeley.com/datasets/dzz48mvjht/1
- Run fetch_dataset.sh to fetch and update the datasets

### Dataset Explained
------------------------------------------------------------------------------------
| Variable Name | Role    | Type        | Demographic |  Units | Missing Values | Description |
| --------------| ------- | ----------- | ----------- | ------ | -------------- | ----------- |
| age           | Feature | Integer     | Age         | years  | no             | Age of the patient |
| sex           | Feature | Categorical | Sex         |        | no             | Sex (1 = male; 0 = female) |
| cp            | Feature | Categorical |             |        | no             | Chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic ) |
| trestbps      | Feature | Integer     |             | mm Hg  | no             | Resting Blood Pressure |
| chol          | Feature | Integer     |             | mg/dl  | no             | Serum Cholestoral |
| fbs           | Feature | Categorical |             |        | no             | Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg       | Feature | Categorical |             |        | no             | Resting electrocardiographic results ( 0 = normal; 1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV); 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria ) |
| thalach       | Feature | Integer     |             |        | no             | Maximum Heart Rate Achieved |
| exang         | Feature | Categorical |             |        | no             | Exercise Induced Angina  (1 = yes, 0 = no) |
| oldpeak       | Feature | Integer     |             |        | no             | ST Depression Induced by Exercise Relative to Rest |
| slope         | Feature | Categorical |             |        | no             | Slope of the peak exercise ST segment  (0 = upsloping; 1 = flat; 2 = downsloping) |
| ca            | Feature | Integer     |             |        | yes            | Number of Major Vessels (0-3) Colored by Flourosopy |
| thal          | Feature | Categorical |             |        | yes            | Thalassemia (0 = error; 1 = normal; 2 = fixed defect; 3 = reversible defect) |
| target        | Target  | Integer     |             |        | no             | Diagnosis of heart disease (1 = present; 0 = absent) |

## Usage
- Run Jupyter Doceker container (Recomended)
```shell
docker run -d -p 3030:8888 --gpus all -v /mnt/localhost/Jupyter/:/home/jovyan --name Jupyter quay.io/jupyter/base-notebook start-notebook.py --NotebookApp.token=''
```
- In Jupyter Terminal install pip packages
```shell
pip install -r requirements.txt
```
- Run Notebook analysis for experimenting

