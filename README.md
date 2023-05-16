# PD-estimation
# Advance PD estimation technique

Here I present the case of 2 main Boosting ensemble techniques to estimate PD (Probability of Default) applied in Credit Risk: 
Adaptive Boosting (AdaBoost) Algoritm and Gradient Boosting. 

The different between 2 algorithms is how they identify the shortcomigs of weak learners (decision trees). 
While Adaboost model identify shortcoming by using high weight for datapoint those difficult to classify and low weight to those easy to classify. 
Gradient Boosting perform the same –grown sequentially, each tree is fitted based on what has been modified from previous tree (a new one is constructed on the residuals of the previous one-which then become the target). Gradient Boosting use gradients in Loss function (y=ax+b+e, e needs a special mention as it is the error term). The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data.

Python notebook using scikit-learn (a machine learning library) provides an implementation of Gradient Boosting ensembles.

# 1.	Gradient Boosting 
Explore Gradient boosting ensemble number of trees effect on performance

```ruby
X_est = X_est.set_index(['date'])
X_est.index = pd.to_datetime(X_est.index)
X_val = X_val.set_index(['date'])
X_val.index = pd.to_datetime(X_val.index)
Y_est = Y_est.set_index(['date'])
Y_est.index = pd.to_datetime(Y_est.index)
```
First, since the DB is expressed in timeseries, set Dataframe index (row labels) using one or more existing columns or array of the correct length corresponding to column date of  X_est, X-val and Y_est. The index can replace the existing index or expand on it.

```ruby
X_est.describe()
Y_est.describe()
X_val.describe()
X_est['V0_L1'].hist()
```
X_est, Y_est, X_val summaries:
X_est: 8 rows x 228 covariates
Y_est: 8 rows x 57 covariates
X_val: 8 rows x 228 covariates
Then we can guess the excel sheet of Y_val should be filled with 8 rows x 57 covariates.
Histogram of X_est shows that datapoints are distributed nearly normal. 

Second, we evaluate a Gradient Boosting algorithm on this dataset using K-fold cross validation:
```ruby
def get_x_cols(y_col,all_x_cols):
    return [i for i in all_x_cols if y_col in i]
```
Define a function get_x_cols to extract covariates i in all x_cols in which also available in y_col.

```ruby
Y_val = pd.DataFrame()
grid = dict()
grid['n_estimators'] = [1,2,3,4,5,10]
grid['learning_rate'] = [0.0001, 0.1, 1.0]
grid['subsample'] = [0.5, 0.7, 1.0]
grid['max_depth'] = [3,4,5]
grid['max_features'] = [None,'auto','log2']
```
Create Grid search space- Grid Search Hyperparameters
- Using a search process to discover a configuration of hyperparameters of the model that works well or best for a given predictive modeling problem. 
- Use the GridSearchCV and specifying a dictionary that maps model hyperparameter names to the values to search. In this case, we will grid search four key parameters for gradient boosting: the number of trees used in the ensemble “n-estimators”, the learning rate, subsample size used to train each tree, and the maximum depth of each tree. We will use a range of popular well performing values for each parameter.

```ruby
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    return scores
```
An evaluate_model function which defines the evaluation procedure. Each configuration combination will be evaluated using repeated k-fold cross-validation with 3 repeats and 10 folds and configurations will be compared using the mean score, in this case, classification accuracy.
Scores variable contains an Array of scores of the estimator for each run of the cross validation

Third, we create a Loop FOR for each covariates (column in Y_est):

```ruby
for y_col in Y_est.columns:
    model = GradientBoostingRegressor()
    x_cols = get_x_cols(y_col,X_est.columns)
    cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=1)
    
    # define the grid search procedure
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')
    # execute the grid search
    grid_result = grid_search.fit(X_est[x_cols], Y_est[y_col])
    best_params = grid_result.best_params_
    # summarize the best score and configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # summarize all scores that were evaluated
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
#     
#     scores = evaluate_model(model, X_est, Y_est[y_col])
#     # fit the model on the whole dataset
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_est[x_cols], Y_est[y_col])
#     results.append(scores)
#     names.append(y_col)
#     # summarize the performance along the way
#     print('>%s %.3f (%.3f)' % (y_col, mean(scores), std(scores)))
#     # make a single prediction
#      row = [[1.20871625,0.88440466,-0.9030013,-0.22687731,-0.82940077,-1.14410988,1.26554256,-0.2842871,1.43929072,0.74250241,0.34035501,0.45363034,0.1778756,-1.75252881,-1.33337384,-1.50337215,-0.45099008,0.46160133,0.58385557,-1.79936198]]
    Y_val[y_col] = model.predict(X_val[x_cols])
```
Inside the loop:
- Extract all covariates of x_est corresponding to y_col. For example  y_col = V0 then x_col = V0_*
- Grid search procedure aims at runing the model with all parameters defined in the grid under condition of minimize scoring='neg_mean_absolute_error' ( minimum of mean of absolute error.
- Then we execute the grid search and append all best parameters in best-params. The result is expressed below showing best score (minimum score of each iteration corresponding to its best parameters of learning rate, max-depth of the tree, max_features, numbers of estimators..)
- Gradient boosting regressor model using those best parameters obtained above and also can use to make prediction. Runing model fit the Gradient Boosting ensemble model for entire dataset in x_est and y_est. The scores then be appended to predict covariates in x_val. The prediction of y_val will be proceeded based on the data in x_val.
	
# 2. AdaBoost Ensemble

(works similar with Gradient boosting except for the library to identify model and parameters of that identify best trees)
```ruby
# evaluate a given model using cross-validation
def get_x_cols(y_col,all_x_cols):
    return [i for i in all_x_cols if y_col in i]
Y_val = pd.DataFrame()
grid = dict()
grid['n_estimators'] = [1,2,3,4,5,10]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    return scores

results, names = list(), list()
for y_col in Y_est.columns:
    model = AdaBoostRegressor()
    x_cols = get_x_cols(y_col,X_est.columns)
    cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=1)
    
    # define the grid search procedure
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')
    # execute the grid search
    grid_result = grid_search.fit(X_est[x_cols], Y_est[y_col])
    best_params = grid_result.best_params_
    # summarize the best score and configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # summarize all scores that were evaluated
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
#     
#     scores = evaluate_model(model, X_est, Y_est[y_col])
#     # fit the model on the whole dataset
    model = AdaBoostRegressor(**best_params)
    model.fit(X_est[x_cols], Y_est[y_col])
#     results.append(scores)
#     names.append(y_col)
#     # summarize the performance along the way
#     print('>%s %.3f (%.3f)' % (y_col, mean(scores), std(scores)))
#     # make a single prediction
# #     row = [[1.20871625,0.88440466,-0.9030013,-0.2268
```
We repeat this process for a specified number of iterations. Subsequent trees help us to classify observations that are not well classified by the previous trees. Predictions of the final ensemble model is therefore the weighted sum of the predictions made by the previous tree models.

If the goal is to classify credit default, then the loss function would be a measure of how good our predictive model is at classifying bad loans. One of the biggest motivations of using gradient boosting is that it allows one to optimise a user specified cost function, instead of a loss function that usually offers less control and does not essentially correspond with real world applications.
One can also clearly observe that the beyond a certain a point (169 iterations for the “cv” method), the error on the test data appears to increase because of overfitting. Hence, our model will stop the training procedure on the given optimum number of iterations.


