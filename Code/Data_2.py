import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random as rnd

# Read data file
headers = ['x', 'y']
df = pd.read_csv('data_2.csv', names=headers)

# Extracting x and y columns
x = df['x'].values
y = df['y'].values

# Converting string to float
for dt in range(0, len(x)):
    x[dt] = float(x[dt])
    y[dt] = float(y[dt])

plt.plot(x, y, '.')

# Method to generate a, b, c parameters of parabola
def calc_parabola_vertex(p, q):
    X = np.vstack([p ** 2, p, np.ones(len(p))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(q)

# Method to do curve fitting, using RANSAC
def use_ransac(df, X, Y):
    dl = len(df)
    ass_prob = 0    # probability that point is inlier
    s = 3           # number of points in a sample
    t = 30          # threshold value

    for i in range(1000):
        pts = rnd.sample(range(0, dl - 1), s)
        a, b, c = calc_parabola_vertex(X[pts], Y[pts])
        # print('Random: ', a, b, c)
        inliers = []
        er = np.absolute(y - (a * X * X) - (b * X) - c)     # error from model to other points
        for j in range(len(er)):
            if er[j] < t:
                inliers.append(X[j])

        new_prob = len(inliers) / len(X)        # probability of inliers

        if new_prob >= ass_prob:
            ass_prob = new_prob
            U, V, W = calc_parabola_vertex(X[pts], Y[pts])
            # print('New: ', a, b, c)
    return U, V, W

# Generating parabola
A, B, C = use_ransac(df, x, y)
y_pos = (A * (x ** 2)) + (B * x) + C

# Plotting Parabola
plt.plot(x, y_pos, linestyle='-', label = 'data_2')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Data 2')
plt.show()
