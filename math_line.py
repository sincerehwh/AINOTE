import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
import math_data 

# import random
# for i in range(0,1000):
# 	print(random.random(),",")


# 参数初始化
w=b=0.0
pre_deta = 100_000
deta = 1000
learning_rate = 0.0001


x_data = [1,2,3,4,5,6,30,8,9]
y_data = [1,2,3,4,5,6,7,8,9]

# x_data = []
# y_data = []
# for i in range(0,50):
# 	x_data.append(math_data.x[i]*100)
# 	y_data.append(math_data.y[i]*100)

print(x_data)
print(y_data)


# 数据矩阵形式
x=np.asarray(x_data)
y=np.asarray(y_data)

print("-----",x)
print("-----",y)

y_plot = []

# 记录轮数
count=0

# 循环更换权重，小于阀值则停止
while abs(pre_deta-deta)>0.01:
	pre_deta = deta
	pre_y = w*x+b
	y_plot.append(pre_y)

	# 误差
	deta = np.sum((y-pre_y)**2)/2
	# 更新权重
	w -= learning_rate*np.sum((pre_y-y)*x)
	b -= learning_rate*np.sum((pre_y-y)*x)

	count += 1
	print(deta,abs(pre_deta-deta))

print("w---:",w)
print("b---:",b)
finial = "完成: y={}*x+({})".format(w,b)
print(finial)

# 可视化
fig = plt.figure()
plt.scatter(x,y,color='',marker="o",edgecolor='g')
line,=plt.plot(x,y_plot[0])

def update(i):
	line.set_ydata(y_plot[i])
	return line

def init():
	line.set_ydata(y_plot[0])

anim = animation.FuncAnimation(fig,update,frames=count,interval=1000,blit=False)
plt.ylim(0,20)
plt.show()





