"""
理论和实验证明，一个两层的ReLU网络可以模拟任何函数。
自行定义一个函数, 使用基于ReLU的神经网络来拟合此函数。
"""
import numpy as np

#选择函数：f(x)=2x+cos(6x) x属于[-100,100]，限定范围便于采样
def final_func(x):
    return 2*x+np.cos(6*x)

#2.搭建网络结构：输入层+隐藏层+输出层，损失函数MSE，优化器Adam
#用前面自己实现的numpy的Matmul和Relu搭建网络
class Matmul:#matrix multiply
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem={'x': x, 'W':W}
        return h
    
    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']
        
        ####################
        '''计算矩阵乘法的对应的梯度'''
        ####################
        #grad_x:dL/dx=grad_y*W^T
        grad_x = np.matmul(grad_y, W.T)
        # grad_W:dL/dW=x^T*grad_y
        grad_W = np.matmul(x.T, grad_y)
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x']=x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        ####################
        x = self.mem['x']
        #grad_x = grad_y * (x > 0).astype(x.dtype) #if x>0 逐元素比较，得到bool数组0/1，再转成和x一样type的数据
        grad_x = np.where(x > 0, grad_y, 0.0) #if x>0,取grad_y，否则取0
        return grad_x
    


class MSE:
    """
    MSE损失计算
    """
    def __init__(self):
        self.mem = {}
        
    def forward(self, y_pred, y_true):
        """
        计算均方误差标量
        y_pred, y_true: 形状相同，一般为 (N, 1) 或 (N, d)
        返回值: 标量 loss
        """
        diff = y_pred - y_true
        self.mem["diff"] = diff
        loss = np.mean(np.square(diff))
        return loss

    def backward(self):
        """
        计算损失对预测值的梯度 dL/dy_pred
        """
        diff = self.mem["diff"]
        numel = diff.size
        grad_y_pred = 2.0 / numel * diff
        return grad_y_pred

class ReLURegressor:
    """两层 ReLU 网络，用于拟合f(x)"""
    def __init__(self, hidden_size=64):
        # 输入维度是1（一元变量），输出维度是1
        self.W1 = np.random.normal(scale=np.sqrt(2.0), size=[1, hidden_size])
        self.b1 = np.zeros([1, hidden_size])

        self.W2 = np.random.normal(scale=np.sqrt(2.0 / hidden_size), size=[hidden_size, 1])
        self.b2 = np.zeros([1, 1])

        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()

    def forward(self, x):
        x = x.reshape(-1, 1)
        h1 = self.mul_h1.forward(x, self.W1) + self.b1  #(N, hidden)
        h1_relu = self.relu.forward(h1) #(N, hidden)

        y_pred = self.mul_h2.forward(h1_relu, self.W2) + self.b2  # (N, 1)

        self.x = x
        self.h1_relu = h1_relu
        self.y_pred = y_pred
        return y_pred

    def backward(self, grad_y):
        #y_pred=h1_relu*W2+b2
        h1_relu_grad, self.W2_grad = self.mul_h2.backward(grad_y)  # (N, hidden)
        self.b2_grad = np.sum(grad_y, axis=0, keepdims=True)       # (1, 1)

        #通过ReLU
        h1_grad = self.relu.backward(h1_relu_grad)                    # (N, hidden)

        #h1=x*W1+b1
        x_grad, self.W1_grad = self.mul_h1.backward(h1_grad)         # (N, 1)
        self.b1_grad = np.sum(h1_grad, axis=0, keepdims=True)        # (1, hidden)
        return x_grad


def train_one_step(model, loss_fn, x, y, lr=1e-3):
    #前向
    y_pred = model.forward(x)
    loss = loss_fn.forward(y_pred, y)

    #反向--先对y_pred取梯度，再反传到各层
    grad_y = loss_fn.backward()
    model.backward(grad_y)

    # 梯度裁剪，防止梯度爆炸
    #np.clip(model.W1_grad, -5.0, 5.0, out=model.W1_grad)
    #np.clip(model.b1_grad, -5.0, 5.0, out=model.b1_grad)

    #np.clip(model.W2_grad, -5.0, 5.0, out=model.W2_grad)
    #np.clip(model.b2_grad, -5.0, 5.0, out=model.b2_grad)

    # 梯度下降更新参数
    model.W1 -= lr * model.W1_grad
    model.b1 -= lr * model.b1_grad
    model.W2 -= lr * model.W2_grad
    model.b2 -= lr * model.b2_grad

    return loss


def evaluate(model, loss_fn, x, y):
    y_pred = model.forward(x)
    loss = loss_fn.forward(y_pred, y)
    return loss, y_pred


#1.采样--训练集+数据集
np.random.seed(42)
#训练集：1000个随机点
x_train = np.random.uniform(-100, 100, 1000).astype(np.float32).reshape(-1, 1)
y_train = final_func(x_train) #噪声：0.05 * np.random.randn(1000, 1).astype(np.float32)

#print(x_train)
#print(y_train)

#测试集：300个均匀点
x_test = np.linspace(-200, -100, 300).astype(np.float32).reshape(-1, 1)#测一下泛化性，采样范围换掉
y_test = final_func(x_test)

#为了训练稳定，对输入和目标做标准化
x_mean, x_std = np.mean(x_train), np.std(x_train) + 1e-8
y_mean, y_std = np.mean(y_train), np.std(y_train) + 1e-8
x_train_n = (x_train - x_mean) / x_std
y_train_n = (y_train - y_mean) / y_std
x_test_n = (x_test - x_mean) / x_std
y_test_n = (y_test - y_mean) / y_std

# 3. 训练--两层ReLU网络拟合f(x)
model = ReLURegressor(hidden_size=64)
loss_fn = MSE()

num_epochs = 1000
learning_rate = 1e-3
batch_size = 64

for epoch in range(1, num_epochs + 1):
    indices = np.random.permutation(x_train_n.shape[0])
    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, x_train_n.shape[0], batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        xb = x_train_n[batch_idx]
        yb = y_train_n[batch_idx]
        batch_loss = train_one_step(model, loss_fn, xb, yb, lr=learning_rate)
        epoch_loss += batch_loss
        num_batches += 1

    loss = epoch_loss / max(1, num_batches)
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch}: train MSE(normalized) = {loss:.6f}")

test_loss_n, y_pred_test_n = evaluate(model, loss_fn, x_test_n, y_test_n)
y_pred_test = y_pred_test_n * y_std + y_mean
test_loss = np.mean((y_pred_test - y_test) ** 2)

print(f"test MSE(normalized): {test_loss_n:.6f}")
print(f"test MSE(original scale): {test_loss:.6f}")