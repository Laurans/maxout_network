import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ------------- DATA ----------------
x_train = np.linspace(-30, 30)
y_train = x_train**2

# Define constants of our maxout units
d_in = 1
d_out = 1
pool_size = 3

# -------------- GRAPH --------------
# Placeholder
x = tf.placeholder(tf.float32, [None, d_in])
labels = tf.placeholder(dtype=tf.float32, shape=[None, d_out])

# Learnable parameters
W = tf.Variable(tf.random_normal([d_in,d_out*pool_size]))
b = tf.Variable(tf.zeros([d_out*pool_size]))

# Computation
z = tf.matmul(x, W) + b
h = tf.reduce_max(tf.reshape(z, [-1,d_in,d_out,pool_size]), axis=3)
h = tf.reshape(h, [-1,1])

# ------------- LEARNING CONTEXT -------------
loss = tf.reduce_sum(tf.square(h-labels))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

# ------------- TF SESSION -------------
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training
for _ in range(5000):
    for batch_xs, batch_ys in zip(x_train, y_train):
        sess.run(train_step, feed_dict={x:[[batch_xs]], labels:[[batch_ys]]})

# ------------- VISUALIZING APPROXIMATION ----------
def graph(formula, x_range):
    x = np.array(x_range)
    x = np.expand_dims(x, 1)
    y = formula(x)
    for i in range(y.shape[-1]):
        plt.plot(x, y[:, i])
    plt.plot(x, quadratique(x))
    plt.show()

def quadratique(x):
    return x ** 2

def maxout_linears(x):
    weights = W.eval()
    bias = b.eval()
    return weights * x + bias

graph(maxout_linears, range(-30,31))