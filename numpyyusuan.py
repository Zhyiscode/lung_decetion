import numpy as np
import numpy as ny


'''
#两层感知机，与门的逻辑操作
def AND(x1,x2):
    w1,w2, theta =0.5,0.5,0.7
    tmp = x1*w1 +x2*w2
    if tmp <= theta:
        return 0
    elif tmp >theta:
        return 1
x1 =int(input('输入x1：'))
x2 =int(input('输入x1：'))
print(f"输入参数为{x1}he{x2}")
shuchu =AND(x1,x2)
print(f"shuchu1jieguo1:{shuchu}")


import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7
y = w*x
print(y)
c = np.sum(y)
print(c)
v = np.sum(y)+b
print(v)
最后加if判断与0做比较，输出0或1.
# 与非门，和或门的逻辑操作只是将权重进行更改
与非门只需将参数变为负值，例如  -0.5，-0.5，-0.7
或门的参数，例如  0.5，0.5，0.3
'''

'''
#三层感知机，去实现异或门的操作
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    temp = np.sum(x*w) +b
    if temp<=0:
        return 0
    else:
        return 1


def NAND(x1,x2):
    x = np.array( [x1, x2] )
    w = np.array( [-0.5, -0.5] )
    b = 0.7
    temp = np.sum( x * w ) + b
    if temp<=0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array( [x1, x2] )
    w = np.array( [0.5, 0.5] )
    b = -0.3
    temp = np.sum( x * w ) + b
    if temp <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 =OR(x1,x2)
    s2 = NAND( x1, x2 )
    s = AND( s1, s2 )
    return s

x1 = int(input("shurucanshu1:"))
x2 = int(input("shurucanshu2:"))
y = XOR(x1,x2)
print(y)
'''

'''
h(x)只在0和1变换的激活函数为上述感知机所用函数。if的判断只适用于浮点数的输入，如果输入为数组类型
x = np.array([-1.0,1.0,2.0])
y=x>0
print(y)
输出为[False  True  True]
输出为bool型
加入y = y.astype(np.int)
def step_function(x):
激活函数sigmoid函数h（x）=1/[1+e(-x)]

'''
x = np.array([-1.0,1.0,2.0])
y=x>0
y = y.astype(np.int)
print(y)