from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

df= pd.read_csv('house_dataset.csv',encoding = "ISO-8859-1")
train_data,test_data=train_test_split(df,train_size=0.8,random_state=3)
y_test=np.array(test_data['price']).reshape(-1,1)

features1=['sqft_above','sqft_basement','sqft_living','sqft_lot','grade']

#X = df[features].values.reshape(-1, len(features))
#y = df[target].values

reg=linear_model.LinearRegression()
reg.fit(train_data[features1],train_data['price'])
pred=reg.predict(test_data[features1])
mean_squared_error=metrics.mean_squared_error(y_test,pred)
print('mean squared error(MSE): ', round(np.sqrt(mean_squared_error),2))
print('R squared training: ',round(reg.score(train_data[features1],train_data['price']),3))
print('R squared test: ', round(reg.score(test_data[features1],test_data['price']),3))
#print('Intercept: ', reg.intercept_)
print('Coefficient: ', reg.coef_)


id = []

for i in range(len(train_data)):
    id.append(i)

id2 = []

for i in range(len(test_data)):
    id2.append(i)


print("\nQueries: \n")
for i in range(20):
    q_p1 = pred[i]
    q_ry1 = y_test[i][0]
    print("Predicted y: ",q_p1)
    print("Real y: ", q_ry1)
    print("\n")

plt.plot(id2,test_data['price'], color='black', label="real y's")
plt.plot(id2,pred, color="blue",label="predicted y's")
plt.title("Framework: Real  vs Predicted: Test")
plt.legend()
plt.show()

tds = sorted(test_data['price'])
preds = sorted(pred)

plt.plot(id2,tds, color='black',label="real y's")
plt.plot(id2,preds, color="blue",label="predicted y's")
plt.title("Framework: Real  vs Predicted: Test (Sorted)")
plt.legend()
plt.show()
