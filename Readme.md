# Heart Disease Predection

## DataSet Source
- [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
```shell
wget -O heart-disease.zip https://archive.ics.uci.edu/static/public/45/heart+disease.zip
```
------------------------------------------------------------------------------------
| Variable Name | Role    | Type        | Demographic |  Units | Missing Values | Description |
| --------------| ------- | ----------- | ----------- | ------ | -------------- | ----------- |
| age           | Feature | Integer     | Age         | years  | no             |             |
| sex           | Feature | Categorical | Sex         |        | no             |             |
| cp            | Feature | Categorical |             |        | no             |             |
| trestbps      | Feature | Integer     |             | mm Hg  | no             | Resting Blood Pressure (on admission to the hospital) |
| chol          | Feature | Integer     |             | mg/dl  | no             | Serum Cholestoral |
| fbs           | Feature | Categorical |             |        | no             | Fasting Blood Sugar > 120 mg/dl |
| restecg       | Feature | Categorical |             |        | no             |             |
| thalach       | Feature | Integer     |             |        | no             | Maximum Heart Rate Achieved |
| exang         | Feature | Categorical |             |        | no             | Exercise Induced Angina |
| oldpeak       | Feature | Integer     |             |        | no             | ST Depression Induced by Exercise Relative to Rest |
| slope         | Feature | Categorical |             |        | no             |             |
| ca            | Feature | Integer     |             |        | yes            | Number of Major Vessels (0-3) Colored by Flourosopy |
| thal          | Feature | Categorical |             |        | yes            |             |
| num           | Target  | Integer     |             |        | no             | Diagnosis of Heart Disease |

