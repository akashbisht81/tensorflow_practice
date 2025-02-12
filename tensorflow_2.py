import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32,name = 'Y')
#############################
########## TO DO ############
#############################

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w = tf.get_variable('w', initializer=tf.constant(0.0))
b = tf.get_variable('b', initializer=tf.constant(0.0))
#############################
########## TO DO ############
#############################

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = (w * X + b)
#############################
########## TO DO ############
#############################

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name = 'loss')
#############################
########## TO DO ############
#############################

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

# Create a filewriter to write the model's graph to TensorBoard
#############################
########## TO DO ############
#############################
writer = tf.summary.FileWriter('/graphs',tf.get_default_graph())


with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    #############################
    ########## TO DO ############
    #############################
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model for 100 epochs
    for i in range(100):
        # run 100 epochs
    		for x, y in data:
    			# Session runs train_op to minimize loss
    			sess.run(optimizer, feed_dict={X: x, Y:y})




        # print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # close the writer when you're done using it
    #############################
    ########## TO DO ############
    #############################
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w,b])
    #############################
    ########## TO DO ############
    #############################

print('Took: %f seconds' %(time.time() - start))
print('weight',w_out)
print('bias',b_out)
print('loss',loss)
# uncomment the following lines to see the plot
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
