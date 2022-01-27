
# In Hospital Mortality Prediction

  

The predictors of in-hospital mortality for intensive care units (ICU)-admitted HF patients remain poorly characterized. We aimed to develop and validate a prediction model for all-cause in-hospital mortality among ICU-admitted HF patients.

  

This project attempts to create a classifier that can identify patients that will die or remain alive based on certain parameters.

  
  

## Source

  

This project is an implementation of the kaggle project found at [https://www.kaggle.com/saurabhshahane/in-hospital-mortality-prediction](https://www.kaggle.com/saurabhshahane/in-hospital-mortality-prediction)

  

## Models
TheRE is one classification model in this project
### 1. `xgboostmodel.py`
This uses the **XGBClassifier**. A summary of the performance of this model is given below
|  |precision|recall|f1-score|support|
|--|--|--|--|--|
|0 (Alive)|0.95|0.96|0.95|204
|1 (Dead)|0.96|0.95|0.95|204
|**accuracy**|||0.95|408
|**macro avg**|0.95|0.95|0.95|408
|**weighted avg**|0.95|0.95|0.95|408