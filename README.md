<img width="428" height="102" alt="image" src="https://github.com/user-attachments/assets/08f58ce0-8583-45cf-9120-f7e6e759ccf7" /># SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: Pandas, scikit-learn modules (SGDRegressor, StandardScaler).

2.Load the dataset and extract input features X (Size, Bedrooms) and target variables Y (Price, Occupants).

3.Normalize the input features using StandardScaler.

4.Initialize two SGD Regressor models for predicting Price and Occupants.

5.Train both models using the scaled input data and respective target variables.

6.Take user input, scale it, predict Price and Occupants, and display the results.. 

## Program:

/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Evangeline Sophiya P M
RegisterNumber:  212225040086
*/
```
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("house.csv")
data.columns=data.columns.str.strip()
x=data[['Size','Bedrooms']]
y_price=data['Price']
y_occ=data['Occupants']
scaler=StandardScaler()
x_scale=scaler.fit_transform(x)
price_model=SGDRegressor(max_iter=1000,learning_rate='constant',eta0=0.01)
occ_model=SGDRegressor(max_iter=1000,learning_rate='constant',eta0=0.01)
price_model.fit(x_scale,y_price)
occ_model.fit(x_scale,y_occ)
size=int(input("Enter the size of the room: "))
bed=int(input("Enter the number of bedrooms: "))
x_new=scaler.transform([[size,bed]])
price=price_model.predict(x_new)
occ=occ_model.predict(x_new)
print("Price: ",price[0])
print("Occupants: ",int(occ[0]))
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)

<img width="428" height="102" alt="Screenshot 2026-05-22 192603" src="https://github.com/user-attachments/assets/e4f18c07-7408-474f-b002-77bf43ca8a50" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
