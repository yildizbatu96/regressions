import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import pandas as pd 
import seaborn as sns
from pylab import rcParams
plt.rcParams['figure.figsize'] = (12,8)



# LINEER REGRESSION 
# LINEER REGRESSION 
# LINEER REGRESSION 
# LINEER REGRESSION 

df = pd.read_csv('bike_sharing_data.txt')
df.head()

ax= sns.scatterplot(x='Population', y='Profit', data=df,)
ax.set_title("Profit in $10000s vs City Population in 10000s")
plt.show()

# Cost Function 𝐽(𝜃)

def costfunc(X, y, theta):
    m=len(y)
    y_pred = X.dot(theta)
    error = (y_pred - y ) ** 2 

    return 1/(2*m) * np.sum(error)

m = df.Population.values.size
X = np.append(np.ones((m, 1)), df.Population.values.reshape(m, 1), axis=1)
y = df.Profit.values.reshape(m, 1)
theta = np.zeros((2,1))


costfunc(X, y, theta)

# Gradient Descent (minimazind the cost function)

def gradesc(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = X.dot(theta)
        error = np.dot(X.transpose(), (y_pred-y))
        theta -= alpha * 1/m * error
        costs.append(costfunc(X, y, theta))
    return theta, costs

theta, costs = gradesc(X, y, theta, alpha=0.01, iterations=2000)
print("h(x) = {} + {}x1".format(str(round(theta[0, 0], 2)),str(round(theta[1,0], 2))))


# Visualizing the Cost Function 𝐽(𝜃)

from mpl_toolkits.mplot3d import Axes3D

theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 100)

costval = np.zeros((len(theta0), len(theta1)))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = np.array([theta0[i], theta1[j]])
        costval[i, j] = costfunc(X, y, t)

fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta0, theta1, costval, cmap = 'viridis')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel("$\Theta0$")
plt.ylabel("$\Theta1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30, 330)
plt.show()


# Plotting the Convergence

plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Values of the Cost Function over Iterations of Gradient Descent")


# Training Data with Linear Regression Fit

theta.shape
theta
theta = np.squeeze(theta)

sns.scatterplot(x='Population', y='Profit', data=df)

x_val = [x for x in range(5,25)]
y_val = [(x * theta[1] + theta[0]) for x in x_val]
sns.lineplot(x_val, y_val)
plt.xlabel("Population of the City in 10000s")
plt.ylabel("Profit of the City in $10.000s")
plt.title("Lineer Regression Fit")
plt.show()

# Inference Using the Optimized 𝜃 Values

def predict(x, theta):
    y_pred = np.dot(theta.transpose(), x)
    return y_pred

y_pred_1 = predict(np.array([1,4]), theta) * 10000
print("Population for 40,000 People the model predicts a profit of $", str(round(y_pred_1, 0)))


y_pred_2 = predict(np.array([1, 8.3]), theta) * 10000
print("Population for 83,000 People the model predicts a profit of $", str(round(y_pred_2, 0)))



# LOGISTIC REGRESSION
# LOGISTIC REGRESSION
# LOGISTIC REGRESSION
# LOGISTIC REGRESSION

data = pd.read_csv('DMV_Written_tests.csv')
data.head()

scores = data[['DMV_Test_1', 'DMV_Test_2']].values
results = data['Results'].values

passed = (results == 1).reshape(100,1)
failed = (results == 0).reshape(100,1)

ax = sns.scatterplot(x=scores[passed[:,0],0], y=scores[passed[:,0],1], marker="^", color='green', s=60) 

ax = sns.scatterplot(x=scores[failed[:,0],0], y=scores[failed[:,0],1], marker="x", color='red', s=30) 

ax.set(xlabel="DMV Written Test 1 Scores", ylabel="DMV Written Test 2 Scores")
ax.legend(['Passed', 'Failed'])
plt.show()

# Defining Logistic Sigmoid Function 𝜎(𝑧)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# We should get 0.5 for input 0
sigmoid(0)

# Computing the Cost Function 𝐽(𝜃) and Gradient

def compute_cost(theta, x, y):
    m = len(y)
    y_predict = sigmoid(np.dot(x, theta))
    errorvalue = (y * np.log(y_predict)) + (1-y)*np.log(1-y_predict)
    cost = -1 / m * sum(errorvalue)
    gradient = 1/m * np.dot(x.transpose(),(y_predict-y))
    return cost[0], gradient

# Cost and Gradient at Initialization

mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)
scores = (scores-mean_scores) / std_scores

rows = scores.shape[0]
cols = scores.shape[1]

X = np.append(np.ones((rows,1)), scores, axis=1)
y = results.reshape(rows, 1)

theta_init = np.zeros((cols+1, 1))
cost, gradient = compute_cost(theta_init, X, y)

print("Cost at initialization", cost)
print("Gradient at initialization", gradient)

def gradient_descent(x, y, theta, alpha, iterations):
    kost = []
    for i in range(iterations):
        cost, gradient = compute_cost(theta, x, y)
        theta -= (alpha * gradient)
        kost.append(cost)
    return theta, kost

theta, kost = gradient_descent(X, y, theta_init, 1, 200)
print("Theta after running gradient descent: ", theta )
print("Resulting cost: ", kost[-1])

# Plotting the Convergence of 𝐽(𝜃)

plt.plot(kost)
plt.xlabel("Iterations")
plt.ylabel("+J(\Theta)$")
plt.title("Values of Cost Function over Iterations of Gradient Descent")


# Plotting the Decision Boundary

ax = sns.scatterplot(x=X[passed[:,0],1], 
                        y=X[passed[:,0],2], 
                        marker="^", color='green', s=60) 

ax = sns.scatterplot(x=X[failed[:,0],1], 
                        y=X[failed[:,0],2], 
                        marker="x", color='red', s=30) 
ax.legend(['Passed', 'Failed'])
ax.set(xlabel="DMV Written Test 1 Scores", ylabel="DMV Written Test 2 Scores")

x_boundary = np.array([np.min(X[:, 1]), np.max(X[0:, 1])])
y_boundary = -(theta[0] + theta[1] * x_boundary) /theta[2]

sns.lineplot(x=x_boundary, y=y_boundary, color='blue')
plt.show()


# Predictions Using the Optimized 𝜃 Values

def pred(theta, x):
    sonuc = x.dot(theta)
    return sonuc > 0

p = pred(theta, X)
print("Training Accuracy: ", sum(p==y)[0],"%")

test = np.array([50, 79])
test = (test - mean_scores)/std_scores
test = np.append(np.ones(1), test)
probab = sigmoid(test.dot(theta))
print("A person who scores 50 and 79 on ther DMV Written Tests have a ", 
np.round(probab[0], 2), " probability of passing")
