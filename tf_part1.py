import tensorflow as tf

a = tf.constant(2,name = 'a')
b = tf.constant(3, name = 'b')
x = tf.add(a, b)

z = tf.zeros([2,3],tf.int32)
s = tf.lin_space(10.0,13.0,10)
r = tf.range(3,18,2)
v = tf.get_variable("scalar", initializer=tf.constant(2))
w = tf.Variable(10)
nw=w.assign(100)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
    print(sess.run(x))
    print(sess.run(z))
    print(sess.run(s))
    print(sess.run(r))
    print(sess.run(tf.global_variables_initializer()))
    print(sess.run(v))
    print(sess.run(w.initializer))
    print(nw.eval())
writer.close()
