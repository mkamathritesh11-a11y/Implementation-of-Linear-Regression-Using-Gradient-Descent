# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Give inputs.
3. Create a model for Gradient scale using linear regression.
4. Predict y using x.
5. provide your customizations for title,xlabel,ylabel,font,colour,legends etc.
6. Note the output.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Ritesh M Kamath
RegisterNumber: 25010798

import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []


for _ in range(epochs):
    y_hat = w * x + b

   
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="green")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")
```

## Output:
<img width="697" height="618" alt="image" src="https://github.com/user-attachments/assets/1541dc92-2d31-4eed-9f70-30bf21c3c947" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
