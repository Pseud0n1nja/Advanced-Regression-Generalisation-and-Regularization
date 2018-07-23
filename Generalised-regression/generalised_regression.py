#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

#fetching data
elec_cons = pd.read_csv("total-electricity-consumption-us.csv",  sep = ',', header= 0 )


#checking NA
elec_cons.isnull().values.any()

#setting parameters
X = elec_cons.Year.reshape(-1,1)
y = elec_cons.Consumption

# Doing Polynomial regression and comparing it with linear, quadratic and cubic fit
for degree in range(1,4):
    pipeline = Pipeline([('poly_features', PolynomialFeatures(degree=degree)),
                     ('model', LinearRegression())
                   ])
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    
# plot predictions and actual values against year
    fig, ax = plt.subplots()
    ax.set_xlabel("Year")                                
    ax.set_ylabel("Power consumption")
    ax.set_title("Degree= " + str(degree))
    ax.scatter(elec_cons.Year, y, color="pink")
    ax.plot(elec_cons.Year, y_pred, color="red")
    plt.show()
