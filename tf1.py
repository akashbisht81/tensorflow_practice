#No more Warning
## WARNING: ? The tensorflow library wasn't compiled to use SSE4.1 instructions,
                # but these are available on your machine and could speed up CPU
                # computation.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

#creating a scalar variable
s = tf.Variable(2,name = 'scalar')

m = tf.Variable([0,1],[2,3], name = 'matrix')

w = tf.Variable(tf.zeros([784,10]))

#evaluate values of variables
v = tf.get_variable("normal_matrix", shape = (784,10), initializer=tf.truncated_normal_initializer())

# initialize all variable

#uncomment this when you feel it-------------------------------------------
# with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # print(v.eval())
#uncomment this when you feel it-------------------------------------------

# where in the case we don't initialize variables  using
    #sess.run(tf.global_variables_initializer()) ??
    #When we use assign
    #W.assign(100) doesn't assign the value 100 to W,
    #but instead create an assign op to do that. For this op to take effect, we have to run this op in session.

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval())
