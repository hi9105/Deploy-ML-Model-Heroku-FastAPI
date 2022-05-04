# Model Card

- Author: Hiral


## Model Details

- Random Forest Classifier model is used. Dataset has around 60,000 records.

- The model predicts that whether a person has salary > 50K or salary <= 50K.

- Find factors that can influence a person's salary.

- Training dataset is 80%, validation and test dataset is 20 % each.

- DVC is used to record pipeline stages and metrics.

- DVC with AWS S3 is used to store artifacts remotely.


## Intended Use

- It can be used to automate CI/CD for Machine Learning projects with FastAPI and Heroku.

- It can be used by any type of agencies like government or private.


## Training Data

- For training, 80 % of dataset have been used.

- All the empty spaces have been removed.

- One hot encoding is used for categorical features.

- Label binarizer is used for the labels.


## Evaluation Data

- For validation, 20 % of dataset have been used.


## Metrics

- For understanding the model performance, metrics like Precision, Recall and F-score have been used.


## Ethical Considerations

- Census data contains important information like : age, sex, marital status, household composition, family characteristics, and household size.

- Sometimes, it can easily have incorrect, misleading and outdated information.

- While making any recommendation or important decision, precautions should be taken.


## Caveats and Recommendations

- This model does not perform Hyper-parameters optimization. It just uses a pre-defined set of parameters. Hyper-parameters can be tuned. 

- More models can also be used and compare the performance.