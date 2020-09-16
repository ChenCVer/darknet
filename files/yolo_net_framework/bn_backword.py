import numpy as np


def batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Input:
	- x: (N, D)维输入数据
	- gamma: (D,)维尺度变化参数 
	- beta: (D,)维尺度变化参数
	- bn_param: Dictionary with the following keys:
	- mode: 'train' 或者 'test'
	- eps: 一般取1e-8~1e-4
	- momentum: 计算均值、方差的更新参数
	- running_mean: (D,)动态变化array存储训练集的均值
	- running_var：(D,)动态变化array存储训练集的方差

	Returns a tuple of:
	- out: 输出y_i（N，D）维
	- cache: 存储反向传播所需数据
	"""
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)

	N, D = x.shape
	# 动态变量，存储训练集的均值方差
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

	out, cache = None, None
	# TRAIN 对每个batch操作
	if mode == 'train':
	sample_mean = np.mean(x, axis = 0)
	sample_var = np.var(x, axis = 0)
	x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
	out = gamma * x_hat + beta
	cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)
	running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	running_var = momentum * running_var + (1 - momentum) * sample_var
	# TEST：要用整个训练集的均值、方差
	elif mode == 'test':
	x_hat = (x - running_mean) / np.sqrt(running_var + eps)
	out = gamma * x_hat + beta
	else:
	raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return out, cache


def batchnorm_backward(dout, cache):
    """
    Inputs:
    - dout: 上一层的梯度,维度(N, D),即 ∂L/∂y
    - cache: 所需的中间变量,来自于前向传播

    Returns a tuple of:
    - dx: (N, D)维的: ∂L/∂x
    - dgamma: (D,)维的: ∂L/∂γ
    - dbeta: (D,)维的: ∂L/∂β
    """
    x, gamma, beta, x_hat, sample_mean, sample_var, eps = cache
    N = x.shape[0]

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma
    dsigma = -0.5 * np.sum(dx_hat * (x - sample_mean), axis=0) * np.power(sample_var + eps, -1.5)
    dmu = -np.sum(dx_hat / np.sqrt(sample_var + eps), axis=0) - 2 * dsigma * np.sum(x - sample_mean, axis=0) / N
    dx = dx_hat / np.sqrt(sample_var + eps) + 2.0 * dsigma * (x - sample_mean) / N + dmu / N

    return dx, dgamma, dbeta
