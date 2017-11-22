"""
We are trying to produce a simple neural network that does
linear regression. Well to begin with, it is maybe worth
stating that this way of doing a linear regression maybe 
is not the best, however, it is a start to taming the 
tensorflow beast. 
(cc) 2017 Ali Rassolie
"""
print("[GLOBAL] importing")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("[GLOBAL] Setting os environ TF_CPP_MIN_LOG_LEVEL to '2' ")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# the function
def f(x):
	return x

x = [[np.random.randint(10), np.random.randint(10)] for i in range(60000)]
y = []

for i in x: 
	if i[1] > f(i[0]):
		y.append(1)
	else: 
		y.append(0) 

x = np.array(x, dtype="float32")
y = np.array(y, dtype="float32")

# print(f"Shape of x: {x[:100].shape}")

"""
So if we look at the list that we provide the placeholder
in the case of the example provided by Martin Gorer, we 
noted that he provided a so called tensor with the following
dimensions [None, 28,28,1] where the None represented the amount
of data input, and the 28x28x1 represented the dimensions of the 
image where the 1 represented the color channel which was only one. 
In our case, we are simply inputting y and x coordinates, pairwise. 
So we will need to use None as well, cuz we dont know how many inputs
we are going to give at one time. the 2 in turn represents the 
dimensions of that input, which with consideration are just 2 values. 
"""
print("[GLOBAL] Starting")
X = tf.placeholder(tf.float32, [None, 2])
# similarly, we will only need two weights, and one node?
W = tf.Variable(tf.zeros([2,1]))
# One bias value for the one node. 
b = tf.Variable(tf.zeros([1]))

# This model is in effect the operation we are going to perform
# in our node of interest. Note that we are in effect constructing
# a graph, which we then are going to drive with the Session.run 
# method later. The model will thus become a vector containing
# the activations of the neuron. 
model  = tf.nn.softmax(tf.matmul(X,W) + b)
# In this case we only have two labels, for each cluster. 
# I is this one that has been causing all of the issues. 
labels = tf.placeholder(tf.float32, [None]) 

# The error function
error = -tf.reduce_sum(labels*tf.log(model))
# So here we select what kind of optimization method we want to use
# and the learning rate in turn. tf.train contains the class. 
# I am unsure whether the 0.003 value represents the amount with
# which we are going to change the weights and the biases. Also,
# does it know automatically what weights it is going to change?
optimizer  = tf.train.GradientDescentOptimizer(0.003)
# Furthermore, the GradientDescentOptimizer has a minimze method
# associated with it, which we would like to feed the error
# function into, which in effect tells it the cost function it 
# should use. 
train_step = optimizer.minimize(error)

init = tf.global_variables_initializer()

sess = tf.Session()
# Now I assue we are running the init because we want 
# to tell Session how the graph looks like. 
sess.run(init)
for i in range(1000):
	train_data = {X: x[(i*100):((i+1)*100)], labels: y[(i*100):((i+1)*100)]}
	
	# Aight, so it seems that we are running the last operation
	# of the graph in order to drive the entire thing, which
	# makes sense. 
	sess.run(train_step, feed_dict=train_data)

print("[GLOBAL] Finished")