#Let's start
import numpy as np
from matplotlib import pyplot as plt

X_train = np.loadtxt('data/x.dat')
y_train = np.loadtxt('data/y.dat')
tau = 0.01

def pred(X_train, y_train, x,tau):
    m = np.size(X_train, 0)
    n = np.size(X_train, 1)
    theta = np.zeros((n, 1))
    y_medial = []
    for i in range(m):
        y = []
        y.append(y_train[i])
        y_medial.append(y)
    y_train = y_medial


    x_medial = []
    x_medial_shuxian = []

    for i in range(m):
        x_medial.append(x)
    #每一横排合成(69,2)变成(69,1)
    x_medial = X_train - x_medial


    # print(np.shape(x_medial))
    for i in range(m):
        x_whatever = []
        # print(x_medial[i][0],x_medial[i][1])
        x_whatever.append(np.power(x_medial[i][0],2)+np.power(x_medial[i][1],2))
        x_medial_shuxian.append(x_whatever)

    # print(np.shape(x_medial_shuxian))
    x_medial = np.array(x_medial_shuxian)
    w = np.exp(- x_medial / (2 * tau))  # .^2表示每一项都进行平方，双竖线就表示向量做差之后各项平方和开根号

    #x = np.repmat(x, m, 1)#手动完成repmat操作
    g = np.ones([n, 1])

    #为什么数值不改变呢
    while (np.linalg.norm(g) > 1e-6):
        h = 1 / (1 + np.exp(- X_train.dot(theta)))#sigmoid 点乘
        g = np.transpose(X_train).dot(w * (y_train - h)) - theta.dot(1e-4)
        medial_whatever = []
        medial_whatever_2 = w * h * (1-h)
        for i in range(m):
            medial_whatever.append(medial_whatever_2[i][0])
        H = -(np.transpose(X_train).dot(np.diag(medial_whatever)).dot(X_train)) - np.eye(n).dot(1e-4)#eye的意思是生成n行n列的单位矩阵

        theta = theta - np.linalg.inv(H).dot(g)

    y_predicted = np.double(np.transpose(x) * theta > 0)
    return y_predicted


iter = 50
result = [[0]*iter for i in range(iter)]
print(result)
for i in range(iter):
    for j in range(iter):
        x = [0, 0]
        x[0] = 2*i/(iter-1) - 1
        x[1] = 2*j/(iter-1) - 1

        print(pred(X_train, y_train, x, tau))
        # result[j][i] = pred(X_train, y_train, x, tau)


    # def loss_plot(self, loss_type):
    #     iters = range(len(self.losses[loss_type]))
    #     plt.figure()
    #     plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
    #     plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
    #     if loss_type == 'epoch':
    #         plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
    #         plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
    #     plt.grid(True)
    #     plt.xlabel(loss_type)
    #     plt.ylabel('acc-loss')
    #     plt.legend(loc="upper right")
    #     plt.show()


plt.figure(1)
plt.clf()
plt.plot((iter/2)*(1+X_train(y_train==0,1))+0.5, (iter/2)*(1+X_train(y_train==0,2))+0.5, 'ko')
plt.plot((iter/2)*(1+X_train+y_train(y_train+y_train==1,1))+0.5, (iter/2)*(1+X_train(y_train==1,2))+0.5, 'kx')
plt.text(iter/2 - iter/7, iter + iter/20, ['tau = '+ tau], 'FontSize', 18);
plt.show()




#Conclusions: