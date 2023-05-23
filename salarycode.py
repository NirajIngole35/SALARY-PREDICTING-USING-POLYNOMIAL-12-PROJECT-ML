import pandas as pd
import numpy as np
ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\SALARY PREDICTING USING POLYNOMIAL 12 PROJECT ML\SALARY_LEVEL.csv")
print(ds.head(4))
print(ds.tail(4))
print(ds.describe)

#find x and y
x=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values
# use PolynomialFeatures
from sklearn .preprocessing import PolynomialFeatures
sc=PolynomialFeatures(degree=4)
x1=sc.fit_transform(x)
#graph
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(x,y,)

plt.xlabel("salary")
plt.show()
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x1,y)
#predict using polynomial
x=6
print(model.predict(sc.fit_transform([[x]])))
