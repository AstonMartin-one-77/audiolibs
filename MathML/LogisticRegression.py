
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic loss function allows to have "bowl" curve with one minimum
# Note: istead of direct derivative of sigmoid function
def loss_logistic_function(X:np.array, y:np.array, w:np.array, b):
    f_x = 1/(1 + np.exp(-(X @ w + b)))
    return -y*np.log(f_x) - (1-y)*np.log(1-f_x)

def cost_logistic_function(X:np.array, y:np.array, w:np.array, b, aggr=0):
    loss = loss_logistic_function(X, y, w, b)
    return np.sum(loss)/len(loss) + np.sum(aggr*w)/(2*len(loss))

def grad_logistic_function(X, y, w, b, aggr=0):
    # X(m,n): m - examples, n - features
    # w(n): n - features
    # b: scalar
    # (X @ w + b) -> vector(m)
    f_x = sigmoid(X @ w + b)
    diff = f_x - y
    dj_dw = (diff @ X + aggr*w)/len(diff)
    dj_db = np.sum(diff)/len(diff)
    return dj_dw, dj_db

def logistic_model_fit(X_t:np.array, y:np.array, w_init:np.array, b_init, lerning_rate, iterations:int, aggr=0):
    cost_stat = []
    w_out = np.copy(w_init)
    b_out = np.copy(b_init)
    for i in range(iterations):
        dj_dw, dj_db = grad_logistic_function(X_t, y, w_out, b_out, aggr)
        w_out -= lerning_rate*dj_dw
        b_out -= lerning_rate*dj_db
        if 0 == i % 1000:
            cost_stat.append(cost_logistic_function(X_t, y, w_out, b_out, aggr))
    cost_stat.append(cost_logistic_function(X_t, y, w_out, b_out, aggr))
    return w_out, b_out, cost_stat

if __name__ == "__main__":
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_init = np.random.randn((X_train.shape[1]))/2
    b_init = np.random.randn((1))/2
    w_out, b_out, cost_stat = logistic_model_fit(X_train, y_train, w_init, b_init, 0.18, 10000, 0.001)
    print(cost_stat)
    print(w_out)
    print(b_out)
