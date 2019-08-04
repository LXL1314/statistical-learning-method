import numpy as np
import matplotlib.pyplot as plt

def train(X, Y, lr=1): ###更新方式为SGD
    
    X_size = X.shape[0]
    alpha = np.zeros([X.shape[0], 1])
    b = 0
    iter_count = 0  # 迭代次数
    print('迭代次数\t误分类点\t\talpha\t\t\t\tb')
    while True:
        tcount = 0  # 分类正确的点
        for i in range(X_size):
            if Y[i]*(b + np.dot(X[i], np.dot(X.T, alpha*Y)))<=0:
                alpha[i] += lr
                b += Y[i]
                iter_count += 1
                print(iter_count, '\t\t\t', 'x' + str(i), '\t\t', alpha.T, '\t\t', b)
                break  # 本次迭代结束,跳出for循环
            tcount += 1
            
        if tcount == X_size:
            break
    return np.dot(X.T, alpha*Y), b

def plot_points(X, y, w, b):
    # 绘制图像
    plt.figure()
    x1 = np.linspace(0, 6, 100)
    x2 = (-b - w[0]*x1)/w[1]
    plt.plot(x1, x2,color='r',)

    for i in range(len(X)):
        if (y[i] == 1):
            plt.scatter(X[i][0],X[i][1],marker='o',color='g',s = 50)
        else:
            plt.scatter(X[i][0],X[i][1],marker='x',color='b',s = 50)
    plt.show()

if __name__ == '__main__':
    X = [[3,3],[4,3],[1,1],[3,2],[2,2],[6,3],[5,3]]
    X = np.array(X)
    y = [1,1,-1,-1,1,-1,1]
    y = np.array(y).reshape([X.shape[0], 1])
    lr = 1 #学习率
    w, b = train(X,y,lr)
    plot_points(X, y, w, b)