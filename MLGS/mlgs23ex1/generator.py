import numpy as np


class HMM_TxtGenerator:
    def __init__(self, corpus, K):
        """Given the set of sentences `corpus` and number of states `K`, builds an HMM.
           Firstly it makes the volcabulary `self.word_list` based on all present words in 
           `corpus`. The variable `self.word_list` is a list of words. Then index of the word
           `self.word_list[v]` is v. Moreover, this function constructs `self.model_params`
           which is an instance of randomly initialized `HMM_Params`.

        Parameters
        ----------
        corpus : A list of sentences. Each sentence is a list of words.  
            We will learn model_params using sentences in `corpus`.
        K: int
           Number of possible states, i.e. Z_t \in {0,...,K-1}


        Returns
        -------
        None :
        """
        self.corpus = corpus.copy()
        self.K = K
        # collect all words ---
        word_dic = {}
        for sent in self.corpus:
            for w in sent:
                if w in word_dic:
                    word_dic[w] = word_dic[w] + 1
                else:
                    word_dic[w] = 1
        self.word_list = [u for u in word_dic.keys()]
        self.word_dic = word_dic
        self.V = len(self.word_list)
        # init params
        self.model_params = HMM_Params(K, len(self.word_list))

    def forwards_backwards(self, sentence_in):
        """Does the forwards-backwards algorithm for an observed list of words
           (i.e. and observed sentence).

        Parameters
        ----------
        sentence_in : a list of T words. Each word is a string.

        Returns
        -------
        alpha : np.ndarray, shape=(T,K)
                alpha(t,k) = Pr(Z_t=k,x[1:t])
        beta  : np.ndarray, shape=(T,K)
                beta(t,k)  = Pr(X_{t+1:T}|Z_t=k)
        log_likelihood  : scalar
                log probability of evidence, Pr(X_{1:T}=sentence_in) 
        """
        # convert sentence to numbers
        x = self.sentence_to_X(sentence_in)

        alpha = self.forwards(x)
        log_likelihood = self.log_likelihood(alpha)
        beta = self.backwards(x)

        return alpha, beta, log_likelihood

    def forwards(self, x):
        """Applies the forwards algorithm for a list of observations

        Parameters
        ----------
        x : list
            a list of word-indices like [50,4,3,20]

        Returns
        -------
        alpha : np.ndarray, shape=(T,K)
                alpha(t,k) = Pr(Z_t=k,x[1:t])
        """
        A = self.model_params.A  # [KxK]
        B = self.model_params.B  # [KxV]
        pi = self.model_params.pi  # [Kx1] ------
        T = len(x)
        K = A.shape[0]

        ### YOUR CODE HERE ###
        alpha = np.zeros((T, K))
        # initialization
        alpha[0, :] = pi.T * B[:, x[0]]
        # recursion
        for t in range(1, T):
            alpha[t, :] = alpha[t - 1, :]@A * B[:, x[t]]
        return alpha

    def log_likelihood(self, alpha):
        """Computes the log-likelihood for a list of observations

        Parameters
        ----------
        alpha : np.ndarray, shape=(T,K)
                alpha(t,k) = Pr(Z_t=k,x[1:t])

        Returns
        -------
        log_likelihood  : scalar
                log probability of observations, Pr(X_{1:T}) 
        """

        ### YOUR CODE HERE ###
        T = alpha.shape[0]
        log_likelihood = np.sum(np.log(np.sum(alpha[T - 1, :])))
        return log_likelihood


    def backwards(self, x):
        """Applies the forwards algorithm for a list of observations

        Parameters
        ----------
        x : list
            a list of word-indices like [50,4,3,20]

        Returns
        -------
        beta  : np.ndarray, shape=(T,K)
                beta(t,k)  = Pr(X_{t+1:T}|Z_t=k)
        """
        A = self.model_params.A  # [KxK]
        B = self.model_params.B  # [KxV]
        T = len(x)
        K = A.shape[0]

        ### YOUR CODE HERE ###
        beta = np.zeros((T, K)) # [TxK]
        # initialization
        beta[T - 1, :] = 1
        # recursion
        for t in range(T - 2, -1, -1):
            beta[t, :] = A @ (B[:, x[t + 1]] * beta[t + 1, :])
        return beta

    def E_step(self, sentence_in):
        """Given one observed `sentence_in`, computes sum_chi(i,j), sum_gamma_x(i,j), gamma_1(k).
           The notations correspond to numerator of lecture slide 67.
           Hint: You can begin by computing alpha and beta as
                    `forwards_backwards(self,sentence_in)`

        Parameters
        ----------
        sentence_in : a list of T words. Each word is a string.
                      You can convert sentence_in to a sequence of word-indices
                      as `x = self.sentence_to_X(sentence_in)`. 

        Returns
        -------
        sum_chi : np.ndarray, shape=(K,K)
             Contains values for sum_chi(i,j), numerator of A(i,j) update on slide 67
        sum_gamma_x : np.ndarray, shape=(K,V)
             Contains values for sum_gamma_x(i,j), numerator of B(i,j) update on slide 67
        gamma_1 : np.ndarray, shape=(K,1)
             Contains values for gamma_1(k), Pi(k) update on slide 67.
        """

        # get A,B ------
        A = self.model_params.A  # [KxK]
        B = self.model_params.B  # [KxV]

        # compute alpha[T,K],beta[T,K]
        alpha, beta, _ = self.forwards_backwards(sentence_in)
        T, K = alpha.shape

        # sentence -> x
        x = self.sentence_to_X(sentence_in)

        # compute gamma
        prop_gamma = alpha * beta  # [T,K]
        gamma = prop_gamma/np.reshape(np.sum(prop_gamma, axis=1), newshape=(T, 1))

        # compute si
        si = np.zeros((T-1, K, K))

        for t in range(T-1):
            i_term = alpha[t, :]
            i_term = np.reshape(i_term, newshape=(K, 1))
            j_term = beta[t+1, :] * B[:, x[t+1]]
            j_term = np.reshape(j_term, newshape=(1, K))
            prop_si_t = np.dot(i_term, j_term) * A  # [KxK]
            si[t, :, :] = prop_si_t/np.sum(prop_si_t)

        # compute sum_chi \in [KxK]
        sum_chi = np.sum(si, axis=0)

        # compute sum_gamma_x \in [KxV]
        V = len(self.word_list)
        indic_x = np.zeros([V, T])
        indic_x[x, [i for i in range(T)]] = 1
        sum_gamma_x = np.dot(indic_x, gamma)  # [VxK]
        sum_gamma_x = np.transpose(sum_gamma_x)  # [KxV]

        # compute gamma_1 \in [Kx1]
        gamma_1 = np.reshape(gamma[0, :] , newshape=(K, 1))

        # ret
        return sum_chi, sum_gamma_x, gamma_1

    def generate_sentence(self, sentence_length):
        """ Given the model parameter,generates an observed
            sequence of length `sentence_length`.
            Hint: after generating a list of word-indices like `x`, you can convert it to
                  an actual sentence as `self.X_to_sentence(x)`

        Parameters
        ----------
        sentence_length : int,
                        length of the generated sentence.

        Returns
        -------
        sent : a list of words, like ['the' , 'food' , 'was' , 'good'] 
               a sentence generated from the model.
        """

        ### YOUR CODE HERE ###

        A = self.model_params.A  # [KxK]
        B = self.model_params.B  # [KxV]
        K = A.shape[0]
        V = B.shape[1]
        pi = self.model_params.pi.squeeze()

        # initial draw depending on pi
        word_ids = []
        x = np.zeros(sentence_length, dtype=int)
        z = np.zeros(sentence_length, dtype=int)
        z[0] = np.random.choice(K, p=pi)
        x[0] = np.random.choice(V, p=B[z[0], :])
        word_ids.append(x[0])

        # iteratively generate words id depending on z in last step
        for t in range(1, sentence_length):
            z[t] = np.random.choice(K, p=A[z[t-1], :])
            x[t] = np.random.choice(V, p=B[z[t], :])
            word_ids.append(x[t])

        sent = self.X_to_sentence(word_ids)
        return sent

    def X_to_sentence(self, input_x):
        """Convert a list of word-indices to an actual sentence (i.e. a list of words).
           To convert a word-index to an actual word, it looks at `self.word_list`.


    Parameters
        ----------
        input_x : a list of integer
                  list of word-indices, like [0,6,1,3,2,...,1]


        Returns
        -------
        sent : a list of words like ['the', 'food', 'was', 'good']
        """
        sent = []
        V = len(self.word_list)
        for u in input_x:
            if u < V:
                sent.append(self.word_list[u])
            else:
                raise ValueError("values of input_x have to be in "
                                 + str([0, V-1]) + ", but got the value " + str(u) + ".")

        return sent

    def sentence_to_X(self, input_sentence):
        """Convert a sentence (i.e. a list of words) to a list of word-indices.
           Index of the word `w` is `self.word_list.index(w)`.


        Parameters
        ----------
        input_sentence : list
                         a list of words like ['the', 'food', 'was', 'good']

        Returns
        -------
        X : list
            a list of word-indices like [50,4,3,20]
        """
        X = []
        for w in input_sentence:
            X.append(self.word_list.index(w))
        return X

    def is_in_vocab(self, sentence_in):
        """Checks if all words in sentence_in are in vocabulary.
           If `sentence_in` contains a word like `w` which is not in `self.word_list`,
           it means that we've not seen word `w` in training set (i.e. `curpus`).
        Parameters
        ----------
        sentence_in : list
                      a list of words like ['the', 'food', 'was', 'good']

        Returns
        -------
        to_ret : boolean
            [We've seen all words in `sentence_in` when training model-params.]
        """
        to_return = True
        for w in sentence_in:
            if w not in self.word_list:
                to_return = False
        return to_return

    def update_params(self):
        """ One update procedure of the EM algorithm.
            - E-step: For each sentence like `sent` in corpus, it firstly computes gammas and chis. 
                    Then, it sums them up to obtain numerators for M-step (slide 67).
            - M-step: normalize values obtain at E-step and assign new values to A, B, pi.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # E-step
        K = self.K
        V = self.V

        corpus_sum_chi = np.zeros([K, K])
        corpus_sum_gamma_x = np.zeros([K, V])
        corpus_gamma_1 = np.zeros([K, 1])

        for sent in self.corpus:
            sent_sum_chi, sent_sum_gamma_x, sent_gamma_1 = self.E_step(sent)
            corpus_sum_chi += sent_sum_chi
            corpus_sum_gamma_x += sent_sum_gamma_x
            corpus_gamma_1 += sent_gamma_1

        # M-step
        A_new = corpus_sum_chi / np.reshape(np.sum(corpus_sum_chi, axis=1),newshape=(K, 1))
        B_new = corpus_sum_gamma_x / np.reshape(np.sum(corpus_sum_gamma_x, axis=1), newshape=(K, 1))
        pi_new = corpus_gamma_1 / np.sum(corpus_gamma_1)
        self.model_params.A = A_new
        self.model_params.B = B_new
        self.model_params.pi = pi_new

    def learn_params(self, num_iter):
        """ Runs update procedures of the EM-algorithm for `num_iter` iterations.

        Parameters
        ----------
        num_iter: int
                  number of iterations.
        Returns
        -------
        history_loglik: list of floats
                `history_loglik[t]` is log-probability of training data in iteration `t`.
        """
        history_loglik = []
        for counter in range(num_iter):
            print("iteration " + str(counter) +\
                  " of " + str(num_iter) , end="\r")
            history_loglik.append(self.loglik_corpus())
            self.update_params()
        return history_loglik

    def loglik_corpus(self):
        """ Computes log-likelihood of the corpus based on current parameters.
        Parameters
        ----------
        None
        Returns
        -------
        loglik: float
                log-likelihood of the corpus based on current parameters.

        """
        loglik = 0
        for sent in self.corpus:
            _, _, loglik_of_sent = self.forwards_backwards(sent)
            loglik += loglik_of_sent
        return loglik

    def loglik_sentence(self, sentence_in):
        """ Computes log-likelihood of `sentence_in` based on current parameters.
        Parameters
        ----------
        sentence_in: a list of words
        Returns
        -------
        loglik_of_sent: float
                        log-likelihood of `sentence_in` based on current parameters.
        """
        # check if all words are in corpus.
        for w in sentence_in:
            if w not in self.word_list:
                return -np.Inf
        _, _, loglik_of_sent = self.forwards_backwards(sentence_in)
        return loglik_of_sent


class HMM_Params:

    def __init__(self, n_states, n_symbols):
        """ Makes three randomly initialized stochastic matrices `self.A`, `self.B`, `self.pi`.

        Parameters
        ----------
        n_states: int
                  number of possible values for Z_t.
        n_symbols: int
                  number of possible values for X_t.

        Returns
        -------
        None

        """
        self.A  = self.rnd_stochastic_mat(n_states, n_states)
        self.B  = self.rnd_stochastic_mat(n_states, n_symbols)
        self.pi = self.rnd_stochastic_mat(1, n_states).transpose()

    def rnd_stochastic_mat(self, I, J):
        """ Retruns a randomly initialized stochastic matrix with shape (I,J).

        Parameters
        ----------
        I: int
           shape[0] of desired matrix.
        J: int
           shape[1] of disired matrix.

        Returns
        -------
        x: np.ndarray
           a rondom stochastic matrix with shape (I,J)

        """
        x = np.full((I, J), (1 / J))
        x = x + (np.random.randn(I, J)*(1.0/(J*J)))
        x = x/np.reshape(np.sum(x, axis=1), newshape=(I, 1))
        return x
