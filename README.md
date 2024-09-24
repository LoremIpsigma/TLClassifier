# README and Usage

From the description given in the email I interepreted the task as a classification project - to predict the probability of a given payment succeeding (marked as Executed). The assumptions made were that payments currently in progress in the dataset were 'junk' and could be discarded as they did not provide any information on whether a payment has succeeded or failed. The data I kept consisted of entries that had a status of 'Executed', 'failed' or 'Failed' leading to a smaller dataset although not significantly smaller (~4k).

Most of the data was categorical in nature, whilst the remaining data consisted of timestamp data. 
I created a duration feature which was the difference between the lastupdatedat_ts
and the createdat_ts features - this provided upper and lower duration limits of successful/unsuccessful transactions. 
I also took the difference between authorizing_at and createdat_ts to see how long the initiation of authorization took, perhaps longer times to get to this stage indicated possible failure.
Lastly I took the difference between authorized_at and authorizing_at in the possiblity that longer durations between the two could provide further insight to the outcome of the payment.
In all cases I replaced the NaT values with NaN values as models I had in mind could not work with NaT types.
The categorical features for the most part had a wide range of categories (~ > 20). The range of categories in addition to the numerical features led me to choose a model at this stage, which happened to be XGBoost as it lends itself to situations like this quite nicely which is why there also isn't any scaling of sorts as XGBoost models are invariant to scale unlike linear regression models for example


The next step was to create a training and test set, I used sklearn's train_test_split function for this, keeping 80% of the data for training.
Its important to note that I didn't see an imbalance in the status flags, there were more successes than failures but not significantly larger so no balance steps were needed for this project. 
The training data was then split again, taking 80% of it and using the remaining 20% as test data for feature selection and hyperparameter tuning.
With my total set of features I then began to perform feature selection using XGBoost's native feature_importances property after training a model on a subset of the training data with default model parameters.
The ROC score at this stage was 0.954 - indicating the features were already quite optimal for the model however too many features can hinder the explainability of a model for documentation purposes so I opted to continue with the selection.
I multiplied the feature_importances and had a cutoff of 1. From this only 5 features remained from the initial 10 (I had dropped the timestamp columns and id column). These 5 were as follows: 'bank_id', 'api_version', 'connectivity_type', 'total_duration' and 'authorized_duration'.

The next step was hyperparameter tuning. For this I used the optuna framework in conjunction with XGBoost.
My objective function returned the ROC score and was optimized to maximize this value. I chose 100 trials for this stage. During optimization I followed the same approach for feature selection in which I used the subset of the training data for training and tested on the remaining 20%.
I then took the best trial and its parameters from the study and retrained a new model - this time with the entire training set not just the subset. 
This model was then used on the unseen test data made at the first split and yielded an ROC of 0.956

In terms of improvements, in an ideal world I'd have more features to work with from the start. I'd also substitue ROC for a custom Gini function or KS function.
The objective function would also return a custom metric to be optimized, with more domain knowledge of the data. Feature selection would be more sophisticated with more features, perhaps an iterative process leading to the top features after removing the lowest performing x% of features after each iteration.
Lastly, I'd investigate the hyperparameter tuning further, zeroing in on the best ranges for specific features.

## Usage

Set up a virtual environment

``` python3 -m venv PaymentClassifier ```

Once the environment is activated, install the modules in the requirements.txt file

``` pip3 install -r requirements.txt ```

Import necessary libraries

``` import xgbooost as xgb ```

``` import pandas as pd ```

Load the data into a pandas DataFrame object:

``` df = pd.read_csv(<data>) ```

Run the data through the preprocess function

``` processed = preprocess(df) ```

Load the model

``` model = xgb.Booster() ```
``` model.load_model('model.json') ```

Create DMatrix for data

``` data = xgb.DMatrix(processed[model.feature_names], enable_categorical=True) ```

Predict

``` y_pred = model.predict(data) ``` 
