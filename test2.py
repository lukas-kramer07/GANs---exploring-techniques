import tensorflow as tf
from keras import backend

input = tf.random.normal([80,1000])

print(backend.mean(input) == tf.reduce_mean(input))

#first implementation
y_true1 = tf.constant([-1]*5)
y_pred1= tf.constant([-3,-4,-5,-6,-7])*-1

y_true2 = tf.constant([1]*5)
y_pred2 = tf.constant([-2,-3,-5,-6,0])

print(tf.reduce_mean(y_pred1*y_true1)+tf.reduce_mean(y_pred2*y_true2))

# second implementation
y_true = y_pred1
y_pred = y_pred2
print(tf.reduce_mean(y_true), tf.reduce_mean(y_pred))
print(-(tf.reduce_mean(y_true)-tf.reduce_mean(y_pred)))