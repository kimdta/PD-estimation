# PD-estimation
# Advance PD estimation technique

Here I present the case of 2 different ensemble methodologies to estimate PD (Probability of Default) applied in Credit Risk.
For each record in the db_01.xlsx, the Y variable is the PD 

AdaBoost Algoritm and Gradient Boosting. The different between 2 algorithms is how they identify the shortcomigs of weak learners (decision trees). 
While Adaboost model identify shortcoming by using high weight for datapoint those difficult to classify and low weight to those easy to classify. 
Gradient Boosting perform the same –grown sequentially, each tree is fit base on what has been modified from previous tree. Except that they use gradients in loss function (y=ax+b+e , e needs a special mention as it is the error term). The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data.

# 1.	Gradient Boosting for Regression
Explore Gradient boosting ensemble number of trees effect on performance
[150] import data.

[150] Running created dataset and summarizes the shape of the input and output components, X_est, Y_est, X_val. Which included summary information about number of observations in each covariates

First, Since our data is expressed in timeseries, set Dataframe index (row labels) using one or more existing columns or array of the correct length  corresponding  to column date of  X_est, X-val and Y_est. The index can replace the existing index or expand on it.

[152] X_est, Y_est, X_val summaries. 
X_est: 8 rows x 228 covariates
Y_est: 8 rows x 57 covariates
X_val: 8 rows x 228 covariates
Then guess the excel sheet of Y_val should be filled with 8 rows x 57 covariates.
Histogram of X_est shows that datapoints are distributed nearly normal. 

[156] Then, we evaluate a Gradient Boosting algorithm on this dataset using K-fold cross validation.
- Define a function get_x_cols to extract covariates i in all x_cols in which also available in y_col.
- Create Grid search space- Grid Search Hyperparameters
Using a search process to discover a configuration of hyperparameters of the model that works well or best for a given predictive modeling problem. 
-	Use the GridSearchCV and specifying a dictionary that maps model hyperparameter names to the values to search. In this case, we will grid search four key parameters for gradient boosting: the number of trees used in the ensemble “n-estimators”, the learning rate, subsample size used to train each tree, and the maximum depth of each tree. We will use a range of popular well performing values for each parameter.

A function evaluate_model which define the evaluation procedure. Each configuration combination will be evaluated using repeated k-fold cross-validation with 3 repeats and 10 folds and configurations will be compared using the mean score, in this case, classification accuracy.
Scores variable contains an Array of scores of the estimator for each run of the cross validation

Secondly, create a Loop FOR for each covariates (column in Y_est):
Regression using gradient boosting
x_cols = get_x_cols(y_col,X_est.columns)
-	Extract all columns (covariate) of x_est corresponding to y_col. For example  y_col = V0 then x_col = V0_*
-	Grid search procedure aim at runing the model with all parameters defined in the grid under condition of minimize scoring='neg_mean_absolute_error' ( minimum of mean of absolute error.
-	. Then we execute the grid search and append all best parameters in best-params. The result is expressed below showing best score ( minimum score of each iteration corresponding to its best parameters of learning rate, max-depth of the tree, max_features, numbers of estimators… 
-	# here are summarize all scores that were evaluated
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
-	Then, create Gradient boosting regressor model using those best parameters obtained above and also can use to make prediction. Runing model fit the Gradient Boosting ensemble model for entire dataset in x_est and y_est. The scores then be appended to predict covariates in x_val. The prediction of y_val will be proceeded based on the data in x_val.

the scikit-learn API to evaluate and use Gradient Boosting ensembles, let’s look at configuring the model
	
# 2. Adaboosting ensemble

(works similar with Gradient boosting except for the library to identify model and parameters of that identify best trees)


We repeat this process for a specified number of iterations. Subsequent trees help us to classify observations that are not well classified by the previous trees. Predictions of the final ensemble model is therefore the weighted sum of the predictions made by the previous tree models.
if our goal is to classify credit defaults, then the loss function would be a measure of how good our predictive model is at classifying bad loans. One of the biggest motivations of using gradient boosting is that it allows one to optimise a user specified cost function, instead of a loss function that usually offers less control and does not essentially correspond with real world applications.
One can also clearly observe that the beyond a certain a point (169 iterations for the “cv” method), the error on the test data appears to increase because of overfitting. Hence, our model will stop the training procedure on the given optimum number of iterations.


