
import tensorflow ad tf 

### 0. 设置初始化参数
l_times = 200000 + 1
l_rate = 0.001
print_step = 200  

### 1. 导入数据

# 获取数据

# 提取数据


# 添加占位符
X = tf.placeholder(tf.float32,shape=[None,8])
Y = tf.placeholder(tf.float32,shape=[None,1])

# 初始化参数（随机）
w = tf.Variables()
b = tf.Variables()


#### 2. 构建目标函数

# 使用sigmoid函数: ft.div()


### 3. 构建损失函数

# 损失函数

# 准确率计算


### 4. 运行图执行训练

