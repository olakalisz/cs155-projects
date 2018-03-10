########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding.s If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)  # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for st, p in enumerate(self.A_start):
            probs[0][st] = p * self.O[st][x[0]]
            seqs[0][st] = str(st)

        for i, x_t in enumerate(x[1:]):
            t = i + 1
            for cur_st in range(self.L):
                max_tr_prob = 0
                for prev_st in range(self.L):
                    tr_prob = probs[t - 1][prev_st] * self.A[prev_st][cur_st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        probs[t][cur_st] = tr_prob * self.O[cur_st][x_t]
                        seqs[t][cur_st] = seqs[t - 1][prev_st] + str(cur_st)

        max_prob = 0
        argmax_prob = None
        for st, p in enumerate(probs[M - 1]):
            if p > max_prob:
                max_prob = p
                argmax_prob = st

        max_seq = seqs[M - 1][argmax_prob]
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)  # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        # alphas[0] = [1. for _ in range(self.L)]
        for st, p in enumerate(self.A_start):
            alphas[1][st] = p * self.O[st][x[0]]
        for i, x_t in enumerate(x[1:]):
            t = i + 2
            for cur_st in range(self.L):
                for prev_st in range(self.L):
                    alphas[t][cur_st] += alphas[t - 1][prev_st] * self.A[prev_st][cur_st] * self.O[cur_st][x_t]
            if normalize:
                norm = sum(alphas[t])
                if norm > 0:
                    alphas[t] = [a / norm for a in alphas[t]]

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)  # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        betas[M] = [1 for _ in range(self.L)]
        for t, x_t in reversed(list(enumerate(x))):
            for cur_st in range(self.L):
                for next_st in range(self.L):
                    betas[t][cur_st] += betas[t + 1][next_st] * self.A[cur_st][next_st] * self.O[next_st][x_t]
            if normalize:
                norm = sum(betas[t])
                if norm > 0:
                    betas[t] = [b / norm for b in betas[t]]
        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        # for l1 in range(self.L):
        #     for l2 in range(self.L):

        A = [[0 for i in range(self.L)] for j in range(self.L)]
        O = [[0 for i in range(self.D)] for j in range(self.L)]

        for y in Y:
            for i in range(len(y) - 1):
                l1 = y[i]
                l2 = y[i + 1]
                A[l1][l2] += 1
        for i, a in enumerate(A):
            l1_norm = sum(a)
            self.A[i] = [p / l1_norm for p in a]

        # Calculate each element of O using the M-step formulas.

        for x, y in zip(X, Y):
            for obs, st in zip(x, y):
                O[st][obs] += 1

        for i, o in enumerate(O):
            l1_norm = sum(o)
            self.O[i] = [p / l1_norm for p in o]

    def unsupervised_learning(self, X, N_iters):

        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        # x = X
        for _ in range(N_iters):
            A = np.zeros((self.L, self.L))  # [[0. for i in range(self.L)] for j in range(self.L)]
            O = np.zeros((self.L, self.D))  # [[0. for i in range(self.D)] for j in range(self.L)]
            A_d = np.zeros(self.L)  # [0. for i in range(self.L)]
            O_d = np.zeros(self.L)  # [0. for j in range(self.L)]
            for x in X:
                M = len(x)
                xi = np.zeros((M, self.L, self.L))
                alphas = np.array(self.forward(x, normalize=True))
                betas = np.array(self.backward(x, normalize=True))
                gamma = alphas * betas
                for t, gamma_t in enumerate(gamma[1:]):
                    gamma_t /= np.sum(gamma_t)
                    O_d += gamma_t
                    if t < M - 1:
                        A_d += gamma_t
                    for i in range(self.L):
                        O[i][x[t]] += gamma_t[i]

                for t in range(1, M):
                    for a in range(self.L):
                        for b in range(self.L):
                            xi[t][a][b] += alphas[t][a] * self.A[a][b] * betas[t + 1][b] * self.O[b][x[t]]
                for xi_t in xi[1:]:
                    xi_t /= np.sum(xi_t)
                    A += xi_t

            A /= A_d[:, None]
            O /= O_d[:, None]
            self.A = A.copy()
            self.O = O.copy()
            
    def unsupervised_learning_start(self, X, N_iters):

        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        # x = X
        for zz in range(N_iters):
            print("Iteration %d" % zz)
            A = np.zeros((self.L, self.L))  # [[0. for i in range(self.L)] for j in range(self.L)]
            O = np.zeros((self.L, self.D))  # [[0. for i in range(self.D)] for j in range(self.L)]
            A_d = np.zeros(self.L)  # [0. for i in range(self.L)]
            O_d = np.zeros(self.L)  # [0. for j in range(self.L)]
            A_start = np.zeros(self.L)
            for x in X:
                M = len(x)
                xi = np.zeros((M, self.L, self.L))
                alphas = np.array(self.forward(x, normalize=True))
                betas = np.array(self.backward(x, normalize=True))
                gamma = alphas * betas
                A_start += gamma[1]
                for t, gamma_t in enumerate(gamma[1:]):
                    gamma_t /= np.sum(gamma_t)
                    O_d += gamma_t
                    if t < M - 1:
                        A_d += gamma_t
                    for i in range(self.L):
                        O[i][x[t]] += gamma_t[i]

                for t in range(1, M):
                    for a in range(self.L):
                        for b in range(self.L):
                            xi[t][a][b] += alphas[t][a] * self.A[a][b] * betas[t + 1][b] * self.O[b][x[t]]
                for xi_t in xi[1:]:
                    xi_t /= np.sum(xi_t)
                    A += xi_t
            A_start /= np.sum(A_start)
            A /= A_d[:, None]
            O /= O_d[:, None]
            self.A = A.copy()
            self.O = O.copy()
            self.A_start = A_start.copy()

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        import numpy as np
        emission = []
        states = []
        state = np.random.choice(self.L, p=self.A_start)
        for _ in range(M):
            states.append(state)
            emission.append(np.random.choice(self.D, p=self.O[state]))
            state = np.random.choice(self.L, p=self.A[state])

        return emission, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(420)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    random.seed(69)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning_start(X, N_iters)

    return HMM
