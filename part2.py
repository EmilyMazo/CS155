#!/usr/bin/python

import nltk.tag
import nltk
import numpy as np
import random
import math
import sys

# This version adds start-of-line, end-of-line, start-of-sonnet, and end-of-sonnet tags
import re

def tokenizeSequences(filename):
    training = []
    training_temp = []
    sonnets = open(filename, "r")
    sonnet_list = []
    #sonnet_list_temp = []
    observations = {}
    counter = 0
    for line in sonnets:
        line = line.strip()
        line = line.strip("(")
        line = line.strip(")")
        line = re.split('\s|[?.,!:;]', line)
        if len(line) == 1:
            if line == ['']:
                if counter <= 1:
                    continue
                else:
                    break
            counter += 1
            #sonnet_list.append("<sonnet>")
            #sonnet_list.append(sonnet_list_temp)
            training.append(training_temp)
            training_temp = [('startofsonnet', '')]
            #sonnet_list_temp = []
            continue
        line.append("endofline")
        new_line = []
        new_line.append("startofline")
        for l in line:
            new_line.append(l)
        for l in new_line:
            l = l.lower()
            if l == '':
                continue
            if l not in observations:
                observations[l] = 1
            #sonnet_list_temp.append(l)
            sonnet_list.append(l)
            training_temp.append((l, ''))
        #sonnet_list.append("</sonnet>")
    training_temp.append(("endofsonnet", ''))
    training.append(training_temp)
    training.remove([])
    #sonnet_list.remove([])
    sonnets.close()
    observations['startofsonnet'] = 1
    observations['startofline'] = 1
    observations['endofsonnet'] = 1
    observations['endofline'] = 1
    observations['.'] = 1
    observations[','] = 1
    observations['?'] = 1
    observations['!'] = 1
    observations[':'] = 1
    observations[';'] = 1
    print training
    return training, sonnet_list, observations.keys(), len(observations.keys())    
 

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
        alpha_sum = {}
        for i in range(self.N):
            alpha_list[i] = np.zeros((self.T, 1))
            Bindex = self.obs.index(self.y[0])
            alpha_list[i][0] = pi[i] * B[i][Bindex] # Initialize the start state probability
        for j in range(self.N):
            for t in range(self.T - 1):
                alpha_col_sum = 0.0 # This will be the sum of all a_i(t) for this t
                alpha_sum[j] = 0.0
                for i in range(self.N):
                    alpha_col_sum += alpha_list[i][t]
                    alpha_sum[j] += alpha_list[i][t] * A[i][j]
                Bindex = self.obs.index(self.y[t + 1])
                if alpha_col_sum == 0:
                    alpha_list[j][t + 1] = 0
                else:
                    alpha_list[j][t + 1] = B[j][Bindex] * alpha_sum[j] / (alpha_col_sum )
        #print alpha_list
        return alpha_list

    def backward(self, A, B, pi):
        '''
            This function calculates beta vectors
            using the backward algorithm.
        '''
        # This will be a dictionary of vectors.
        # beta_i(t) will be beta_list[i][t].
        beta_list = {}
        beta_sum = {}
        for i in range(self.N):
            beta_list[i] = np.zeros((self.T, 1))
            beta_list[i][self.T - 1] = 1.0
        for j in range(self.N):
            for t in range(self.T - 2, -1, -1):
                beta_col_sum = 0.0
                beta_sum[j] = 0.0
                for n in range(self.N):
                    beta_col_sum += beta_list[n][t + 1]
                    Bindex = self.obs.index(self.y[t + 1])
                    beta_sum[j] += beta_list[n][t + 1] * A[j][n] * B[n][Bindex]
                if beta_col_sum == 0 :
                    beta_list[j][t] = 0
                else: 
                    beta_list[j][t] = beta_sum[j] / (beta_col_sum )
        #print beta_list
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
        delta = np.empty([self.N, self.N, self.T])
        delta_sum = {}
        for t in range(self.T - 1):
            delta_sum[t] = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    Bindex = self.obs.index(self.y[t + 1])
                    num = alpha_list[i][t] * A[i][j] * beta_list[j][t + 1] * B[j][Bindex]
                    delta_sum[t] += num                    
                    delta[i][j][t] = num 
        for i in range(self.N):
            for j in range(self.N):
                for t in range(self.T - 1):
                    delta[i][j][t] = delta[i][j][t] / delta_sum[t] 
        #for x in range(self.T):
        #    temp = 0.0
        #    for m in range(self.N):
        #        for n in range(self.N):
        #            temp += delta[m][n][t]
        #    print "temp"
        #    print temp
        return delta

    def update_A(self, gamma, delta):
        '''
            This function creates a new, updated transition matrix.
        '''
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            sumDenom = 0.0
            for e in range(self.T-1):
                sumDenom += gamma[i][e]
            for j in range(self.N):
                sumNum = 0.0
                for t in range(self.T - 1):
                    sumNum += delta[i][j][t]
                A[i][j] = sumNum / sumDenom
        #print delta
        print abs(sum(A[0]))
        #assert abs(sum([A[0][x] for x in range(self.N)]) - 1) < 0.001
        print A
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
                for t in range(self.T):
                    if (self.y[t] == self.obs[k]):
                        sumNum += gamma[i][t]
                    sumDenom += gamma[i][t]
                B[i][k] =  (sumNum / sumDenom)
        print abs(sum(B[0]))
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
        A_sum = np.zeros((self.N, 1))
        B_sum = np.zeros((self.N, 1))
        pi_sum = 0.0
        for n in range(self.N):
            for i in range(self.N):
                A[n][i] = np.random.gamma(1,1)
                A_sum[n] += A[n][i]
            for k in range(self.K):
                B[n][k] = np.random.gamma(1,1)
                B_sum[n] += B[n][k] 
            pi[n] = np.random.gamma(1,1)
            pi_sum += pi[n]
            for j in range(self.N):
                A[n][j] = A[n][j] / A_sum[n]
            for e in range(self.K):
                B[n][e] = B[n][e] / B_sum[n]
        for m in range(self.N):
            pi[m] = pi[m] / pi_sum
        #for i in range(self.N):
        #    print "first B"
        #    print sum(B[i])
        is_converged = False
        counter = 0
        while (is_converged != True):
            print ("new iteration")
            #self.y = random.choice(Y)
            self.y = Y
            self.T = len(self.y)
            alpha_list = self.forward(A, B, pi)
            #print "done alpha"
            beta_list = self.backward(A, B, pi)
            #print "done beta"
            gamma = self.get_gamma(alpha_list, beta_list)
            #print "done gamma"
            delta = self.get_delta(A, B, alpha_list, beta_list)
            #print "done delta"
            Anew = self.update_A(gamma, delta)
            #print "a new"
            #print Anew
            Bnew = self.update_B(gamma)
            #print "b new"
            pinew = self.update_pi(gamma)
            #print "pi new"
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

def stopping(log_diff, num):
    if num > 25:
        return True
    else:
        return False

if __name__ == '__main__':
    em = EM()
    training, sonnets, obs, k = tokenizeSequences("willy shakes and spenser.txt")
    em.N = 20 # This can be set to whatever we want it to be
    em.K = k 
    print k
    em.obs = obs
    #hmm = em.train(sonnets)
    state_num = 27
    states = range(state_num)
    observations = list(obs)
    w = open("AO.txt", "w")
    hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states, observations)
    hmm = hmm_trainer.train_unsupervised(training, max_iterations=200)
    print hmm._outputs[1]._samples # print this just to see what the words are
    outputs = np.zeros((state_num, k))
    w.write("O")
    w.write("\n")
    for i in range(state_num):
        line = hmm._outputs[i]._data
        for l in range(len(line)):
            #print -1.0 * line[l]
            line[l] = math.exp(line[l])
        outputs[i] = line
        w.write(str(line))
        w.write("\n")
    print outputs
    w.write("A \n")
    w.write("\n")
    transitions = np.zeros((state_num, state_num))
    for s in range(state_num):
        line = hmm._transitions[s]._data
        for l in range(len(line)):
            line[l] = math.exp(line[l])
        transitions[s] = line
        w.write(str(line))
        w.write("\n")
    # The probabilities are all hugely negative. Not sure what kind of distribution that is
    # or if we should normalize it somehow.
    print transitions
    print "max output"
    print np.amax(outputs)
    coords_list = []
    # this finds the coordinates of the five most negative elements in outputs
    for j in range(state_num):
        for m in range(10):
            max = np.amax(outputs[j])
            #max = np.amin(outputs[j])
            coords = np.argwhere(outputs[j] == max)
            x = coords[0][0]
            coords_list.append((j, x))
            #outputs[j][x] = -1 * sys.maxint
            outputs[j][x] = 0.0
    fifth_smallest_element = max
    # coords_list is a list of fifth smallest elements in outputs
    print coords_list        
    # use coords_list to find the emissions associated with these transitions
    learn = open("learnedStates.txt", "w")
    for c in coords_list:
        token = hmm._outputs[c[0]]._samples[c[1]]
        #print nltk.pos_tag([token]), c[0], outputs[c[0]][c[1]]
        learn.write(str(c[0]))
        learn.write("\n")
        learn.write(str(token))
        learn.write("\n")
    w.close()
    learn.close()




























