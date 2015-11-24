import numpy as np
from numpy.linalg import inv, det


class CTM:
    # counts is an array of num_docs x vocab_size, where the element is the number of counts of that word.
    def __init__(self, num_docs, num_topics, vocab_size, counts):
        self.K = num_topics
        self.num_docs = num_docs
        self.lambdas = np.zeros(num_topics)
        self.nus_squared = np.zeros(num_topics)
        self.zeta = 0.001
        self.phi = np.ones((vocab_size, num_topics))
        self.mu = np.zeros(num_topics)
        self.sigma = np.diag(np.ones(num_topics))
        self.sigma_inv = inv(self.sigma)
        self.beta = np.ones((num_topics, vocab_size))
        self.counts = counts
        self.vocab_size = vocab_size

    # Updates zeta using equation 14
    def update_zeta(self):
        self.zeta = np.sum(np.exp(self.lambdas + self.nus_squared/2))

    # For a given document, update phi
    #def update_phi(self, doc_index)


	# This is based on equations 8-12 from CTM paper
	def bound(self, doc_index, lamdas, nu_squared_vals):
		N = np.sum(self.counts[doc_index, :])
		total_bound = 0
		total_bound += .5*np.log(det(self.sigma_inv))
		total_bound -= (self.K/2)*np.log(2*np.pi)
		# Equation 9
		total_bound -= .5*(np.trace(np.diag(nu_squared_vals).dot(self.sigma_inv)) + (lamdas - self.mu).T.dot(self.sigma_inv.dot(lamdas - self.mu)))
		total_bound += N*((1./(self.zeta))*(np.sum(np.exp(lamdas + nu_squared_vals/2))) + 1 - np.log(self.zeta))

		# n is the word_id, k are the topic id
		# This contains the terms of equations 10, 11, 12 that are dependent on each word-topic pair (sum_n sum_k)
		for k in xrange(self.K):
			for n in xrange(self.vocab_size):
				total_bound += counts[doc_index, n]*self.phi[n, k]*(lamdas[k] + np.log(self.beta[k, n]) - np.log(self.phi[n, k]))
		# Contains the part of equation 11 not dependent on word-topic pairs
		total_bound += .5*(np.sum(np.log(nu_squared_vals))) + (self.K/2)*np.log(2*np.pi) + self.K/2
		return total_bound

	def full_bound(self):
		return np.sum(bound(doc, self.lambdas, self.nus_squared) for doc in xrange(self.num_docs))






ctm = CTM(4, 2, 3, [[1, 2, 3], [2, 1, 3], [2, 2, 2], [1, 4, 5]])
print ctm.K
print ctm.lambdas
print ctm.nus_squared
print ctm.zeta
print ctm.phi

x = np.array([1,2,3])



