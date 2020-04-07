# importing the numpy, pandas and matplotlib for carrying out tasks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reads the data from excel using pandas
data_1= pd.read_csv("data_1.csv")
# prints the number of rows and columns in the format(rows, format)
print(data_1.shape)  
# prints the first five rows
print(data_1.head()) 

# calls all the values of x and y from the excel
X = data_1['x'].values
Y = data_1['y'].values
n = 250 # Total Number of values in X or Y

# converts the data in the form of a matrix [1 X X^2] and [Y]
x = np.column_stack([np.ones(n),X,X**2])
y = np.transpose(np.array(Y))

xt = np.transpose(x) #finding the transpose of matrix x
xtx = np.dot(xt,x) #finding the dot matrix multiplication between xt and x
xty = np.dot(xt,y) #finding the dot matrix multiplication between xt and y

#Solving for a in [xtx]*a = [xty]
a = np.linalg.solve(xtx,xty)

# print (a)

# Calculating the values of x and y
xp = np.linspace(0,500,50)
yp = a[0] + a[1]*xp + a[2]*xp**2

# print(xp)
# print(yp)

# Ploting the scatter points
plt.plot(X,Y,'ko',label='data_1')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Data 1')
    
# Plotting the regression non linear line
plt.plot(xp,yp,linewidth=5)
plt.show()    
    