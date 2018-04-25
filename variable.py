#!/usr/bin/python
#  -*- coding: UTF-8 -*-
import tensorflow as tf

with tf.variable_scope('foo'):
    v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))
print(v)

with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))
print(v)

with tf.variable_scope('root'):
    print(tf.get_variable_scope().reuse)    # False

    with tf.variable_scope('foo', reuse=True):
        print(tf.get_variable_scope().reuse)    # True

        with tf.variable_scope('bar'):
            print(tf.get_variable_scope().reuse)    # True

    print(tf.get_variable_scope().reuse)    # False

with tf.variable_scope('foo_root'):
    with tf.name_scope('foo1'):
        # v = tf.get_variable('v1', [1], initializer=tf.constant_initializer(1.0))
        v = tf.Variable(tf.constant(1.0, tf.float32, shape=[1]), name='v1')
        print(v)
    with tf.name_scope('foo2'):
        v = tf.Variable(tf.constant(2.0, tf.float64, shape=[1]), name='v2')
        print(v)