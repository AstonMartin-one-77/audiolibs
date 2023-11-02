
import numpy as np


# Gradient Descent for linear regression:
#       X[m,n] - m - number of examples, n is number of features
#       w[n] - weights of linear regression
#       y[m] - target outputs
#       dj_dw[j] = (np.dot(X[i,:],w)+b - y[i])*X[j], j 0...n-1
#       dj_db = (np.dot(X[i,:],w)+b - y[i])
def linear_gradient_descent(X:np.array, y:np.array, w:np.array, b):
    m = X.shape[0]
    n = X.shape[1]
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        diff = np.dot(X[i,:], w) + b - y[i]
        for j in range(n):
            dw[j] += diff*X[i,j]
        db += diff
    return dw/m, db/m

def linear_cost(X:np.array, y:np.array, w:np.array, b):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        cost += (np.dot(X[i,:], w) + b - y[i])**2
    return cost/(2*m)

def linear_scale(X:np.array):
    mu = np.sum(X, axis=0)/X.shape[0]
    diff = X - mu
    std = np.sqrt(np.sum(diff**2,axis=0)/X.shape[0])
    return diff/std

def linear_model_fit(X_t:np.array, y:np.array, w_init:np.array, b_init, lerning_rate, iterations:int):
    cost_stat = []
    cost_stat.append(linear_cost(X_t, y, w_init, b_init))
    w_out = np.copy(w_init)
    b_out = np.copy(b_init)
    for i in range(iterations):
        dj_dw, dj_db = linear_gradient_descent(X_t, y, w_out, b_out)
        w_out -= lerning_rate*dj_dw
        b_out -= lerning_rate*dj_db
        if 0 == i % 100:
            cost_stat.append(linear_cost(X_t, y, w_out, b_out))
    cost_stat.append(linear_cost(X_t, y, w_out, b_out))
    return w_out, b_out, cost_stat

if __name__ == "__main__":
    X_t = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_t = np.array([460, 232, 178])
    b_opt = 785.1811367994083
    w_opt = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    w_init = np.zeros_like(w_opt)
    b_init = 0.0
    w_out, b_out, stat = linear_model_fit(X_t,y_t,w_init,b_init,5.0e-7,1000)
    print(stat)
    print(w_out)
    print(b_out)
    for i in range(X_t.shape[0]):
        print(f"prediction: {np.dot(X_t[i], w_out) + b_out:0.2f}, target value: {y_t[i]}")

    X_norm = linear_scale(X_t)
    w_norm, b_norm, stat = linear_model_fit(X_norm,y_t,w_init,b_init,1.0e-1,1000)
    print(stat)
    print(w_norm)
    print(b_norm)
    for i in range(X_t.shape[0]):
        print(f"prediction: {np.dot(X_norm[i], w_norm) + b_norm:0.2f}, target value: {y_t[i]}")


