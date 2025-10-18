# SC5002-Assignment2_Group4
Assignment 2 SC5002
SLEEP, HEALTH AND LIFESTYLE
Required imports:
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


Data Preprocessing:
1. Reading the Dataset - 
Reads the CSV dataset into a DataFrame named df.
encoding='latin1' handles special characters.
df.head() displays the first 5 rows for a quick look.


df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", encoding='latin1')
df.head()


2. Handling the missing values - 
For numerical columns, missing values are replaced with the mean.
For categorical columns, missing values are replaced with the most frequent value (mode).
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())


for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])


3. Splitting the Dataset
X = all input features except Person ID (irrelevant) and Stress Level (target).
y = target variable (Stress Level).
Splits into training (65%) and testing (35%) sets for model evaluation.
random_state=42 ensures reproducibility.
X = df.drop(['Person ID', 'Stress Level'], axis = 1)
y = df['Stress Level']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42


4. Identifying Categorical and Numerical columns and creating a preprocessor
Builds a transformation pipeline:
Applies OneHotEncoder to categorical columns (turns text into numeric dummy variables).
Passes numerical columns unchanged (they’ll be scaled next).
handle_unknown='ignore' prevents errors if unseen categories appear in test data.
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])
5. Transforming Data and Scaling Numerical Features
preprocessor.fit_transform() → fits encoders and transforms the training data.
preprocessor.transform() → applies the same transformation to test data.
After encoding, data may become sparse (many zeros), so with_mean=False avoids
centering (which breaks sparse matrices).
StandardScaler → standardizes all features (mean=0, std=1) so no feature dominates due to scale differences.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
scaler = StandardScaler(with_mean=False)  # with_mean=False because of sparse matrix
X_train_scaled = scaler.fit_transform(X_train_processed)




Model Fitting
Linear regression: 
1.Train the model using our train dataset:

#train the model using train dataset
linreg = LinearRegression()           Define the model
linreg.fit(X_train_scaled, y_train)   Fit the model to our data


2. Calculate the predicted values of y using X input values based on the model, which will later be used to calculate the residuals (errors). We do so for both training and testing set, such that we can compare between them:


#predicted values of the y training dataset
y_train_pred = linreg.predict(X_train_scaled) 


#apply yhe model to the test data set
y_test_pred = linreg.predict(X_test_scaled)


3.Check the goodness of fit of the Linear regression model by calculating the R^2 values and MSE. By comparing between the training and testing set, we can see whether our model overfits

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Linear Regression Model Train Dataset")
print("Explained Variance (R^2):", linreg.score(X_train_scaled, y_train))
print("Mean Squared Error (MSE):", mean_squared_error(y_train, y_train_pred))
print()


# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Linear Regression Model Test Dataset")
print("Explained Variance (R^2):", linreg.score(X_test_scaled, y_test))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_test_pred))
print()

Ridge Regression:
To test the effect of different alpha values, we used a for loop with 9 iterations, corresponding to alpha(regularization strength) = 1, 2, 3, 4, 5, 6, 7, 8, and 9 respectively. Each iteration trains the model with the same training dataset, and calculates R^2 and MSE values for the training and testing datasets.




for x in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    ridge_modelx = Ridge(alpha = x)  # alpha is the regularization strength
    ridge_modelx.fit(X_train_scaled, y_train)


    # Make predictions on the train set
    y_train_pred = ridge_modelx.predict(X_train_scaled)


    # Make predictions on the test set
    y_test_pred = ridge_modelx.predict(X_test_scaled)


    #Check the goodness of fit of Ridge regression models with different alpha values:
    print('Ridge model' ,x,':', 'alpha =', x)
    # Check the Goodness of Fit (on Train Data)
    print('Train')
    print("Explained Variance (R^2):", ridge_modelx.score(X_train_scaled, y_train))
    print("Mean Squared Error (MSE):", mean_squared_error(y_train, y_train_pred))
    print()


    # Check the Goodness of Fit (on Test Data)
    print('Test')
    print("Explained Variance (R^2):", ridge_modelx.score(X_test_scaled, y_test))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_test_pred))
    print()



K-fold Cross Validation:
1. The X datasets are split before processing. Therefore, we need to recombine the training and testing sets for cross validation. On the other hand, y is not processed, but to get a y dataset that corresponds to X for every data point, we need to recombine y datasets as well.

#Recombine X and y datasets
from scipy.sparse import vstack
X_full = vstack([X_train_scaled, X_test_scaled])
y_full = pd.concat([y_train, y_test], axis=0)


2. 5-fold cross validation: 
Split the dataset into 5 folds, with 1 fold being the testing dataset and the other 4 being the training set for every iteration. Therefore, a 5-fold cross validation gives us 5 iterations and 5 R^2 values for each model. By calculating the average values and standard deviation of the 5 R^2 values, we can compare and determine which model has better fit or more consistent performance. The ridge model with alpha = 9 is chosen, which will be explained in the later section.


#k-fold
kf = KFold(n_splits = 5, shuffle = True, random_state = 42) #5-fold validation
linear_scores = cross_val_score(linreg, X_full, y_full, cv = kf, scoring = 'r2')
ridge_scores  = cross_val_score(ridge_modelx, X_full, y_full , cv = kf, scoring = 'r2')


# Calculate the mean and standard deviation of R² values for each model
print("Linear Regression: Mean R² =", np.mean(linear_scores), " | Standard deviation =", np.std(linear_scores))
print("Ridge Regression : Mean R² =", np.mean(ridge_scores),  " | Standard deviation =", np.std(ridge_scores))


#If ridge regression has larger average R2 and smaller standard deviation, it
#means Ridge gives slightly better and more stable performance — the regularization helps reduce overfitting and variance.














Results analysis:
Linear Regression
High R^2 on both train (0.99) and test (0.88) showing that the model captures strong linear relationships and can explain most of the variation in the data. MSE is small in both cases and shows that predictions are generally close to their actual values.
However, there’s a drop in R² from training (0.99) to test (0.88). This suggests slight overfitting where the model performs a bit better on known data than unseen data. The gap is not big, so the model still generalizes well.








Ridge Regression
As the alpha value increases, Training R^2 slightly decreases as regularisation limits model flexibility. Test R^2 increases and difference of R^2 values between train and test set as overfitting reduces. MSE also decreases on the test set showing a better generalisation.
Since alpha = 9 gives us the closest values of R^2 for the training and testing dataset out of all the Ridge Models, we will use ridge model 9 for cross validation.


K-fold Cross Validation
We can see that the Ridge regression has better and more stable performance because it has higher values and lower standard deviations for its R² score




Evaluations:

Advantages of Linear Regression:
1. It is faster because less amount of computation is needed. Unlike models like Ridge, Lasso, or non-linear regressions, linear regression does not require iterative optimization. It also requires no hyperparameter tuning, and assumes a straight simple line relationship. 

2. It is easier to interpret. In linear regression:

y = w1*x1 + w2*x2 + w3*x3+...+wn*xn
Each coefficient(w) directly tells us the direction of the effect (positive/negative) and the magnitude of the effect(weightage). In comparison, Ridge regression coefficients are shrunken due to regularization, so interpreting the size of coefficients directly becomes less straightforward.
Disadvantages of Linear Regression:
1. Overfitting is very common because no bias is manually introduced to the model, especially when the data is noisy.

2. It is very sensitive to multicollinearity, which is when two or more predictors (independent variables) are highly correlated, meaning they carry overlapping information. This often results in overly large coefficients. The model will still fit the training data well, but the coefficients will fluctuate wildly between different samples, which results in poor generalization.



Advantages of Ridge Regression:
Ridge regression adds a penalty to large coefficients, which helps control model complexity and reduces overfitting, especially when there’s multicollinearity among predictors.
When independent variables are highly correlated, ridge regression stabilizes coefficient estimates, making the model more reliable.


Disadvantages of Ridge Regression:
Unlike Lasso regression, Ridge keeps all predictors in the model (coefficients shrink but don’t become zero), so it doesn’t simplify the model.
The penalty term introduces bias into the estimates while this reduces variance, it can make predictions less accurate if the penalty is too strong.

Application scenarios:

Linear Regression:
Predicting house prices:
Used to estimate property prices based on features like area, number of rooms, and location.
Sales forecasting:
Helps businesses predict future sales based on advertising spend, season, or economic trends.
Ridge Regression:
Medical or genetic data analysis:
In cases with many correlated variables (like gene expressions), ridge regression improves model stability and accuracy.
Predicting stock prices or financial risk:
Used when predictors (like market indicators) are highly correlated — ridge helps prevent overfitting.

Future improvement:
1. Introduce more evaluation metrics besides just R^2 and MSE. We can use MAE(Mean Absolute Error), which is less sensitive to outliers, or RMSE (Root Mean Squared Error), which provides better interpretability in the same scale as our y variable.

2. In Ridge regression, we can introduce more refined hyperparameter tuning such as RidgeCV or GridSearchCV to find the optimal alpha value for the best performance.

3. We can explore other models and compare them to decide the most suitable one. For example, we can explore Lasso Regression for feature selection, or non-linear models like Random Forest Regressor or Gradient Boosting Regressor.




Individual contributions:
JAIN AADITAH: data preprocessing, encoding categorical variables, scaling numerical variables, splitting the data set, evaluation of ridge regression,applications for linear and ridge regression

TEY YI CHING: Calculate R^2 and MSE values for linear and ridge regression, comparing the models and analysing the result, K-fold results analysis, evaluation of ridge regression

DU YUYANG: model fitting, cross validation, evaluation of linear regression, future improvement

