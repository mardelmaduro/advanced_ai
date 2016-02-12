'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Modified: Jacky Baltes
'''

import tensorflow as tf
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

MW = 2.0
Mb = 0.5

yScale = 1000

# Training Data
train_X = numpy.linspace(0, 1, 10)
train_Y = yScale * MW * train_X + numpy.random.randn(*train_X.shape) * 0.33 + Mb

#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.mul(X, W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Testing example, as requested (Issue #2)
    test_X = numpy.linspace(0,1,10) + numpy.random.randn(*train_X.shape) * 0.05
    test_Y = yScale * MW * train_X + Mb

    count = 0
    
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)
    
            fig=plt.figure(figsize=(10,10), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_ylim(0,yScale * MW)
            #ax.set_aspect('equal')
            ax.plot(train_X, train_Y, 'ro', label='Original data')
            ax.plot(test_X, test_Y, 'bo', label='Testing data')
            ax.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
            ax.legend()
            #plt.show()
            fig.savefig('plots/plot_{:05d}.png'.format(count), bbox_inches='tight', dpi=100)
            count = count + 1
            plt.close(fig)
    
        
    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    print "Testing... (L2 loss Comparison)"
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation-Y, 2))/(2*test_X.shape[0]),
                            feed_dict={X: test_X, Y: test_Y}) #same function as cost above
    print "Testing cost=", testing_cost
    print "Absolute l2 loss difference:", abs(training_cost - testing_cost)

    #Graphic display
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_X, train_Y, 'ro', label='Original data')
    ax.plot(test_X, test_Y, 'bo', label='Testing data')
    ax.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    ax.legend()
    #plt.show()
    fig.savefig('plot.png', bbox_inches='tight')