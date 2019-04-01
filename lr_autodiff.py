import autodiff as ad
import numpy as np


def logistic_prob(_w):
    def wrapper(_x):
        return 1 / (1 + np.exp(-np.sum(_x * _w)))
    return wrapper


def test_accuracy(_w, _X, _Y):
    prob = logistic_prob(_w)
    correct = 0
    total = len(_Y)
    for i in range(len(_Y)):
        x = _X[i]
        y = _Y[i]
        p = prob(x)
        if p >= 0.5 and y == 1.0:
            correct += 1
        elif p < 0.5 and y == 0.0:
            correct += 1
    print("总数：%d, 预测正确：%d" % (total, correct))


def plot(N, X_val, Y_val, w, with_boundary=False):
    import matplotlib.pyplot as plt
    for i in range(N):
        __x = X_val[i]
        if Y_val[i] == 1:
            plt.plot(__x[1], __x[2], marker='x')
        else:
            plt.plot(__x[1], __x[2], marker='o')
    if with_boundary:
        min_x1 = min(X_val[:, 1])
        max_x1 = max(X_val[:, 1])
        min_x2 = float(-w[0] - w[1] * min_x1) / w[2]
        max_x2 = float(-w[0] - w[1] * max_x1) / w[2]
        plt.plot([min_x1, max_x1], [min_x2, max_x2], '-r')

    plt.show()


def gen_2d_data(n):
    x_data = np.random.random([n, 2])
    y_data = np.ones(n)
    for i in range(n):
        d = x_data[i]
        if d[0] + d[1] < 1:
            y_data[i] = 0
    x_data_with_bias = np.ones([n, 3])
    x_data_with_bias[:, 1:] = x_data
    return x_data_with_bias, y_data


def auto_diff_lr():
    x = ad.Variable(name='x')
    w = ad.Variable(name='w')
    y = ad.Variable(name='y')

    # 注意，以下实现某些情况会有很大的数值误差，
    # 所以一般真实系统实现会提供高阶算子，从而减少数值误差

    h = 1 / (1 + ad.exp(-ad.reduce_sum(w * x)))
    L = y * ad.log(h) + (1 - y) * ad.log(1 - h)
    w_grad, = ad.gradients(L, [w])
    executor = ad.Executor([L, w_grad])

    N = 100
    X_val, Y_val = gen_2d_data(N)
    w_val = np.ones(3)

    plot(N, X_val, Y_val, w_val)
    executor = ad.Executor([L, w_grad])
    test_accuracy(w_val, X_val, Y_val)
    alpha = 0.01
    max_iters = 300
    for iteration in range(max_iters):
        acc_L_val = 0
        for i in range(N):
            x_val = X_val[i]
            y_val = np.array(Y_val[i])
            L_val, w_grad_val = executor.run(feed_dict={w: w_val, x: x_val, y: y_val})
            w_val += alpha * w_grad_val
            acc_L_val += L_val
        print("iter = %d, likelihood = %s, w = %s" % (iteration, acc_L_val, w_val))
    test_accuracy(w_val, X_val, Y_val)
    plot(N, X_val, Y_val, w_val, True)


if __name__ == '__main__':
    auto_diff_lr()




