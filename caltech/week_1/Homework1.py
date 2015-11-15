# coding: utf-8

#1 D
#2 A
#3 D
#4 B
#5 C
#6 D
#7
# A perceptron learning tool that demonstrates learning on D=2 dataset
# For Homework #1 from "Learning from Data" / Professor Yaser Abu-Mostafa, Caltech
# http://work.caltech.edu/homework/hw1.pdf
# Questions 7-10 can be answered with this implementation

# In this problem, you will create your own target function f and data set D to see
# how the Perceptron Learning Algorithm works. Take d = 2 so you can visualize the
# problem, and assume X = [−1, 1] × [−1, 1] with uniform probability of picking each
# x ∈ X .
#
# In each run, choose a random line in the plane as your target function f (do this by
# taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the
# line passing through them), where one side of the line maps to +1 and the other maps
# to −1. Choose the inputs xn of the data set as random points (uniformly in X), and
# evaluate the target function on each xn to get the corresponding output yn.
#
# Now, in each run, use the Perceptron Learning Algorithm to find g. Start the PLA
# with the weight vector w being all zeros (consider sign(0) = 0, so all points are initially
# misclassified), and at each iteration have the algorithm choose a point randomly
# from the set of misclassified points. We are interested in two quantities: the number
# of iterations that PLA takes to converge to g, and the disagreement between f and
# g which is P[f(x) != g(x)] (the probability that f and g will disagree on their classification
# of a random point). You can either calculate this probability exactly, or
# approximate it by generating a sufficiently large, separate set of points to estimate it.
#
# In order to get a reliable estimate for these two quantities, you should repeat the
# experiment for 1000 runs (each run as specified above) and take the average over
# these runs.
#

# Date: 13/10/15
# Author: Jonathan Becktor

from random import random
import numpy as np
from matplotlib.pylab import *


def generateData(n):
    """
    generates a 2D linearly separable dataset with n samples.
    The thired element of the sample is the label
    """
    x = (np.random.rand(n) * 2 -1 )
    y = (np.random.rand(n) * 2 -1 )
    inputs=[]
    #eq = y=x4+0
    a=[-0.2, -0.8]
    b=[0.25, 1]
    inputs.extend([[x[i], y[i], 1] for i in range(n)])
    for xy in inputs:
        xp=(b[0]-a[0])*(xy[1]-a[1]) - (b[1]-a[1])*(xy[0]-a[0])

        if xp>0:
            xy[2]=1
        else:
            xy[2]=-1

    return inputs

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super().__init__()
        self.w = [0,0] # weights


    def response(self, x):
        """perceptron output"""
        #y = x[0] * self.w[0] + x[1] * self.w[1] # dot product between w and x
        y = sum([i * j for i, j in zip(self.w, x)]) # more pythonic
        if y >= 0:
            return 1
        else:
            return -1

    def updateWeights(self, x):
        """
        updates the weights
        """
        self.w[0] +=  x[2] * x[0]
        self.w[1] +=  x[2] * x[1]
       # print(self.w[0])


    def train(self, data):
        """
        trains all the vector in data.
        """
        learned = False
        iteration = 0
        miss=[0]
        while not learned:
            globalError = 0.0
            miss=[]

            for x in data: # for each sample

                r = self.response(x)
                if x[2] != r: # if have a wrong response
                   # print("%i %i  %i"%(r,x[2],cntr))    #print(data.index(x))
                    miss.append(data.index(x))

            iteration += 1
            if len(miss)==0: # stop criteria
               # print ('iterations: %s' % iteration)
                learned = True # stop learning
                return iteration
            else:
                v2=np.random.randint(len(miss))
                var = miss[v2]
                self.updateWeights(data[var])
               ### print('sturk')
        return 0




iteration_sum = 0
#trainset = generateData(10) # train set generation
#p = Perceptron() # use a short
    #print(p.w)
#iteration_sum+=p.train(trainset)

#print(p.w)
for x in range(1000):
    trainset = generateData(100) # train set generation
    p = Perceptron() # use a short
    #XSprint(x)
    iteration_sum+=p.train(trainset)


print(iteration_sum/1000)
testset = generateData(100) # test set generation
#Perceptron test
for x in testset:
    r = p.response(x)
    if r != x[2]: # if the response is not correct
        print ('not hit.')
    if r == 1:
            plot(x[0], x[1], 'ob')
    else:
            plot(x[0], x[1], 'or')

# plot of the separation line.
# The centor of line is the coordinate origin
# So the length of line is 2
# The separation line is orthogonal to wX
n = norm(p.w) # aka the length of p.w vector
ww = p.w / n # a unit vector
ww1 = [ww[1], -ww[0]]
ww2 = [-ww[1], ww[0]]
print(p.w)
print(p.w/n)
plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
plot([-0.2,0.25],[-0.8,1])
show()
#8
#9
