import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

#fetching Data
car_price = pd.read_csv("/Users/Arpit/Documents/UpGrad/Advanced_Regression/carPrice.csv",  sep = ',', header= 0 )

#na handling
car_price.isnull().values.any()
car_price.info()

# Assigning input and output
X = car_price.drop("price", axis = 1)
y = car_price.price

#Preprocessing the data 
num_attributes = ['symboling', 'wheelbase', 'carlength', 'carheight', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
cat_attributes = ['carCompany', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
      
        
num_pipeline = Pipeline([
				('selector', DataFrameSelector(num_attributes)),
				('scaler', StandardScaler())
			])

cat_pipeline = Pipeline([
				('selector', DataFrameSelector(cat_attributes)),
				('label_encoder', OrdinalEncoder()),
				('one_hot_encoder', OneHotEncoder())  # avoid this step if too much categories in a column
			])

full_pipeline = FeatureUnion(transformer_list=[
					("num_pipeline", num_pipeline),
					("cat_pipeline", cat_pipeline)
			])

full_pipeline.fit(X)
X = full_pipeline.transform(X)

#creating a training set and testing set
#divide the data into 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)


#performing cross validation
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

#assigning alpha some values
params = {'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]}

# ridge model
model = Ridge()

# cross validation
model_cv = GridSearchCV(estimator = model, param_grid = params, scoring= 'neg_mean_absolute_error', cv = folds, verbose = 1)            
model_cv.fit(X_train, y_train)                  

#getting best score
model_cv.best_score_

#getting the optimum value for the alpha
model_cv.best_params_

#complete result of the model
model_cv.cv_results_

#calculating the coefficeints
model_cv.best_estimator_.coef_.shape

type(model_cv.best_estimator_)
