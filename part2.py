#!/usr/bin/python

#import nltk
import numpy as np
import random
import math

# The following code reads each sonnet from a text file.
# Each sonnet is input as a single element into sonnet_list.
# Each line of each sonnet is prepended with <s> and appended with </s>.
# There is currently no other preprocessing.

def tokenizeSequences(filename):
    sonnets = open(filename, "r")
    sonnet_list = {}
    observations = {}
    counter = 0
    for line in sonnets:
        line = line.strip()
        line = line.split(' ')
        if len(line) == 1:
            if line == ['']:
                continue
            counter += 1
            sonnet_list[counter] = []
            continue
        #line.append("</s>")
        new_line = []
        #new_line.append("<s>")
        for l in line:
            if l not in observations:
                observations[l] = 1
            sonnet_list[counter].append(l)
    sonnets.close()
    return sonnet_list, len(observations), observations.keys()
 

class EM(object):
    
    def __init__(self):
        self.N = 0 # The number of hidden states
        self.K = 0 # The number of unique observations
        self.y = [] # The sequence we are training on (list of tokens)
        self.T = 0 # The length of the sequence
        self.obs = [] # The list of unique observations

    def forward(self, A, B, pi):
        '''
            This function calculates alpha vectors 
            using the forward algorithm.
        '''
        # This will be a dictonary of vectors. 
        # alpha_i(t) will be alpha_list[i][t].
        alpha_list = {}
        for i in range(self.N):
            alpha_list[i] = np.zeros((self.T, 1))
            Bindex = self.obs.index(self.y[0])
            alpha_list[i][0] = pi[i] * B[i][Bindex] # Initialize the start state probability
        for j in range(self.N):
            for t in range(self.T - 1):
                alpha_col_sum = 0.0 # This will be the sum of all a_i(t) for this t
                alpha_sum = 0.0
                for i in range(self.N):
                    alpha_col_sum += alpha_list[i][t]
                    alpha_sum += alpha_list[i][t] * A[i][j]
                Bindex = self.obs.index(self.y[t + 1])
                if alpha_col_sum == 0:
                    alpha_list[j][t + 1] = 0
                else:
                    alpha_list[j][t + 1] = B[j][Bindex] * alpha_sum / (alpha_col_sum )
        return alpha_list

    def backward(self, A, B, pi):
        '''
            This function calculates beta vectors
            using the backward algorithm.
        '''
        # This will be a dictionary of vectors.
        # beta_i(t) will be beta_list[i][t].
        beta_list = {}
        for i in range(self.N):
            beta_list[i] = np.zeros((self.T, 1))
            beta_list[i][self.T - 1] = 1.0
        for j in range(self.N):
            for t in range(self.T - 2, -1, -1):
                beta_col_sum = 0.0
                beta_sum = 0.0
                for n in range(self.N):
                    beta_col_sum += beta_list[n][t + 1]
                    Bindex = self.obs.index(self.y[t + 1])
                    beta_sum += beta_list[n][t + 1] * A[j][n] * B[n][Bindex]
                if beta_col_sum == 0 :
                    beta_list[j][t] = 0
                else: 
                    beta_list[j][t] = beta_sum / (beta_col_sum )
        return beta_list

    def get_gamma(self, alpha_list, beta_list):
        '''
            This function calculates the gamma vectors
            for the maximization step of the Baum-Welch algorithm.
        '''
        # This will be a dicitonary of vectors. 
        # gamma_i(t) will be gamma_list[i][t].
        gamma_list = {}
        for m in range(self.N):
            gamma_list[m] = np.zeros((self.T, 1))
        for t in range(self.T):
            gamma_sum = 0.0
            for j in range(self.N):
                gamma_sum += alpha_list[j][t] * beta_list[j][t]
            for i in range(self.N):
                num = alpha_list[i][t] * beta_list[i][t]
                gamma_list[i][t] = num / gamma_sum
                assert gamma_list[i][t] > 0
            temp = 0.0
            for n in range(self.N):
                temp += gamma_list[n][t]
            assert abs(temp - 1) < 0.0001
        return gamma_list

    def get_delta(self, A, B, alpha_list, beta_list):
        '''
            This function calculates the matrix delta 
            for the maximization step of the Baum-Welch algorithm.
        '''
        delta_sum = 0.0
        delta = np.empty([self.N, self.N, self.T])
        for k in range(self.N):
            delta_sum += alpha_list[k][self.T - 1]
        for i in range(self.N):
            for j in range(self.N):
                for t in range(self.T - 1):
                    Bindex = self.obs.index(self.y[t + 1])
                    num = alpha_list[i][t] * A[i][j] * beta_list[j][t + 1] * B[j][Bindex]
                    delta[i][j][t] = num / delta_sum
        for x in range(self.T):
            temp = 0.0
            for m in range(self.N):
                for n in range(self.N):
                    temp += delta[m][n][t]
            print "temp"
            print temp
        return delta

    def update_A(self, gamma, delta):
        '''
            This function creates a new, updated transition matrix.
        '''
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                sumNum = 0.0
                sumDenom = 0.0
                for t in range(self.T - 1):
                    sumNum += delta[i][j][t]
                    sumDenom += gamma[i][t]
                A[i][j] = sumNum / sumDenom
        assert abs(sum([A[0][x] for x in range(self.N)]) - 1) < 0.001
        return A

    def update_B(self, gamma):
        '''
            This function creates a new, updated observtion matrix.
        '''
        B = np.zeros((self.N, self.K))
        for i in range(self.N):
            for k in range(self.K):
                sumNum = 0.0
                sumDenom = 0.0
                for t in range(self.T - 0):
                    if (self.y[t] == self.obs[k]):
                        sumNum += gamma[i][t]
                    sumDenom += gamma[i][t]
                B[i][k] =  1e-50 + (sumNum / sumDenom)
        print B
        return B

    def update_pi(self, gamma): 
        '''
            This function updates the intial state probabilities. 
        '''
        pi = np.zeros((self.N, 1))
        for i in range(self.N):
            pi[i] = gamma[i][0]
        return pi


    def train(self, Y):
        '''
            This function uses Expectation maximization (using the
            Baum-Welch algorithm) to train a Hidden Markov Model.
        '''
        # Initialize A, B, pi to random.
        # Each row (or column, in the case of pi) must sum to 1.
        A = np.zeros((self.N, self.N))
        B = np.zeros((self.N, self.K))
        pi = np.zeros((self.N, 1))
        for n in range(self.N):
            for i in range(self.N):
                A[n][i] = 1.0 / self.N
            for k in range(self.K):
                B[n][k] = 1.0 / self.K 
            pi[n] = 1.0 / self.N
        is_converged = False
        counter = 0
        while (is_converged != True):
            print ("new iteration")
            self.y = random.choice(Y)
            self.T = len(self.y)
            alpha_list = self.forward(A, B, pi)
            beta_list = self.backward(A, B, pi)
            gamma = self.get_gamma(alpha_list, beta_list)
            delta = self.get_delta(A, B, alpha_list, beta_list)
            Anew = self.update_A(gamma, delta)
            Bnew = self.update_B(gamma)
            pinew = self.update_pi(gamma)
            counter += 1
            if counter == 1:
                Adiff = Anew - A
                Bdiff = Bnew - B 
                Anorm = np.linalg.norm(Adiff, 2)
                Bnorm = np.linalg.norm(Bdiff, 2)
                Aconverge = Anorm * 0.1
                Bconverge = Bnorm * 0.1
                print "a converge"
                print Aconverge
                print "b converge"
                print Bconverge
            print "a diff norm"
            print np.linalg.norm(Anew - A, 2)
            print "b diff norm"
            print np.linalg.norm(Bnew - B, 2)
            if np.linalg.norm(Anew - A, 2) < Aconverge:
                if np.linalg.norm(Bnew - B, 2) < Bconverge:
                    is_converged = True
            A = Anew
            B = Bnew
            pi = pinew

if __name__ == '__main__':
    em = EM()
    Y, k, obs = tokenizeSequences("shakespeare.txt")
    em.N = 20 # This can be set to whatever we want it to be
    em.K = k
    em.obs = obs
    hmm = em.train(Y)

        





























