import tensorflow as tf

#creating a scalar variable
s = tf.Variable(2,name = 'scalar')

m = tf.Variable([0,1],[2,3], name = 'matrix')

w = tf.Variable(tf.zeros([784,10]))

#evaluate values of variables
v = tf.get_variable("normal_matrix", shape = (784,10), initializer=tf.truncated_normal_initializer())

# initialize all variable

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v.eval())
