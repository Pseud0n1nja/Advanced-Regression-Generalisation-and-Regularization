import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

#fetching Data
car_price = pd.read_csv("/Users/Arpit/Documents/UpGrad/Advanced_Regression/carPrice.csv",  sep = ',', header= 0 )

#na handling
car_price.isnull().values.any()
car_price.info()

#creating a training set set and testing set

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


#divide the data into 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)


#performing cross validation

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

params = {'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]}

# ridge model
model = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = model, param_grid = params, scoring= 'neg_mean_squared_error', cv = folds, verbose = 1)            
model_cv.fit(X_train, y_train) 
coefficients = model.fit(X_train, y_train).coef_   # extract coefficients for each feature
#plt.plot(range(len(names)), coefficients)          # variable importance

y_pred = model.predict(X_test)
model.score(X_test, y_test)                 

model_cv.best_score_
model_cv.best_params_
model_cv.cv_results_
model_cv.best_estimator_.coef_
type(model_cv.best_estimator_)
