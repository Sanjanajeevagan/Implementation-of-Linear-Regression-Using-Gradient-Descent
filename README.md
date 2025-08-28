# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Import libraries – Load numpy, pandas, and StandardScaler for math, data, and scaling.
2. Define function – Write gradient descent–based linear regression function.
3. Load dataset – Read "50_Startups.csv" using pandas.
4. Extract features/target – Separate X (independent) and y (dependent variable).
5. Scale data – Normalize X and y using StandardScaler.
6. Train model – Call linear_regression to compute theta values.
7. Prepare new input – Reshape and scale new data for prediction.
8. Predict output – Compute dot product with theta and inverse transform.
9. Display results – Print scaled and final predicted value.


## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("C:\\Users\\admin\\Desktop\\50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")


```

## Output:
<img width="373" height="906" alt="Screenshot 2025-08-28 135221" src="https://github.com/user-attachments/assets/8ede5e2e-e6cf-43a3-838e-e3db6bda749a" />

<img width="466" height="819" alt="Screenshot 2025-08-28 135231" src="https://github.com/user-attachments/assets/4135f167-5207-43fb-8c12-92cafe145c2a" />

<img width="455" height="327" alt="Screenshot 2025-08-28 135238" src="https://github.com/user-attachments/assets/e0e2b961-2162-472e-aca9-02d12fd355d9" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
