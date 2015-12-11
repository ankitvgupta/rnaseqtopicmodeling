##############################
# A Serial implementation of CTM
# See CTMParallel.py for better, parallelized version
#
# Author: Ankit Gupta
###########################


import numpy as np
from numpy.linalg import inv, det
from scipy.optimize import fmin_cg, minimize, check_grad
import sys

class CTM:
	# counts is an array of num_docs x vocab_size, where the element is the number of counts of that word.
	def __init__(self, num_docs, num_topics, vocab_size, counts, max_iters, convergence_cutoff):
		self.K = num_topics
		self.num_docs = num_docs
		self.num_topics = num_topics
		self.vocab_size = vocab_size
		self.mu = np.zeros(num_topics)
		self.sigma = np.diag(np.ones(num_topics))
		self.sigma_inv = inv(self.sigma)
		self.beta =  np.random.uniform(0, 1, (num_topics, vocab_size))
		self.saved_lambdas = np.zeros((num_docs, num_topics))
		self.reset_variational_parameters()
		
		self.counts = counts
		
		self.max_iters = max_iters
		self.convergence_cutoff = convergence_cutoff

	def reset_variational_parameters(self):
		self.lambdas = np.random.uniform(0, 1, self.num_topics)#np.zeros(self.num_topics)
		self.nus_squared = np.ones(self.num_topics)
		self.phi = 1.0/self.num_topics*np.ones((self.vocab_size, self.num_topics))
		self.update_zeta()

	# Updates zeta using equation 14
	def update_zeta(self):
		self.zeta = np.sum(np.exp(self.lambdas + self.nus_squared/2))

	def getWordCount(self, doc_index):
		return np.sum(self.counts[doc_index, :])
	# For a given document, update phi for each word
	def update_phi(self, doc_index):
		for word in xrange(self.vocab_size):
			if self.counts[doc_index, word] == 0:
				continue
			phi_tmp = np.exp(self.lambdas) * self.beta[:, word]
			# Normalize this
			phi_tmp /= np.sum(phi_tmp)
			self.phi[word, :] = phi_tmp

	# Uses equation 16
	def update_lambda(self, doc_index):
		def obj(lam):
			return self.bound(doc_index, lam, self.nus_squared)
		# This is a direct implementation of equation 16
		def derivative(lam):
			N = self.getWordCount(doc_index)
			tmp = (-self.sigma_inv.dot(lam - self.mu)) 
			tmp += np.sum(self.counts[doc_index, word_index]*self.phi[word_index, :] for word_index in xrange(self.vocab_size))
			tmp -= (N/self.zeta)*np.exp(lam + self.nus_squared/2)
			return tmp

		# As the paper says, now we can use a gradient based minimizer to minimize this. So, we can use BFGS
		#res = fmin_cg(lambda x: -obj(x), self.lambdas, fprime=lambda x: -derivative(x), full_output=True)
		#print res, '\n'
		opts = {         
			'disp' : False,    # non-default value.
			#'gtol' : 1e-12
		}
		#print self.lambdas
		res = minimize(lambda x: -obj(x), self.lambdas, jac=lambda x: -derivative(x), method='Newton-CG', options=opts)
		#print res
		self.lambdas = res.x

		#for rep in range(100):
		#	print check_grad(lambda x: -obj(x), lambda y: -derivative(y), np.random.uniform(0, .0002, 3))
	
	def update_nu_squared(self, doc_index):
		def obj(nu_sq):
			return self.bound(doc_index, self.lambdas, nu_sq)
		def derivative(nu_sq):
			N = self.getWordCount(doc_index)
			grad = np.zeros(self.K)
			grad += -.5*np.diag(self.sigma_inv)
			grad += -N/(2.*self.zeta)*np.exp(self.lambdas + nu_sq/2) 
			grad += .5/nu_sq
			return grad
		opts = {         
			'disp' : False,    # non-default value.
			}
		bounds = [(0., None) for i in range(self.K)]
		res = minimize(lambda x: -obj(x), self.nus_squared, jac=lambda x: -derivative(x), method='TNC', bounds=bounds, options=opts)
		self.nus_squared = res.x
		#for rep in range(100):
		#	print check_grad(lambda x: -obj(x), lambda y: -derivative(y), np.random.uniform(1., 10.0002, 3))

	def full_bound(self):
		return np.sum(self.bound(doc_id, self.lambdas, self.nus_squared) for doc_id in xrange(self.num_docs))
	# This is based on equations 8-12 from CTM paper
	def bound(self, doc_index, lamdas, nu_squared_vals):
		sys.stdout.flush()
		N = self.getWordCount(doc_index)
		total_bound = 0.0
		total_bound += .5*np.log(det(self.sigma_inv))
		total_bound -= (self.K/2)*np.log(2*np.pi)
		# Equation 9
		total_bound -= .5*(np.trace(np.dot(np.diag(nu_squared_vals), self.sigma_inv)) + (lamdas - self.mu).T.dot(self.sigma_inv.dot(lamdas - self.mu)))
		total_bound += N*(-1./self.zeta*(np.sum(np.exp(lamdas + nu_squared_vals/2))) + 1 - np.log(self.zeta))

		# n is the word_id, k are the topic id
		# This contains the terms of equations 10, 11, 12 that are dependent on each word-topic pair (sum_n sum_k)
		for k in xrange(self.K):
			for n in xrange(self.vocab_size):
				total_bound += self.counts[doc_index, n]*self.phi[n, k]*(lamdas[k] + np.log(self.beta[k, n]) - np.log(self.phi[n, k]))
		# Contains the part of equation 11 not dependent on word-topic pairs
		total_bound += .5*(np.sum(np.log(nu_squared_vals))) + (self.K/2)*np.log(2*np.pi) +  self.K/2
		return total_bound
	def update_variational_parameters(self, doc_index):
		before_bound = self.bound(doc_index, self.lambdas, self.nus_squared)
		for it in xrange(self.max_iters):
			print "     Updating Zeta"
			sys.stdout.flush()
			self.update_zeta()
			print "     Updating Lambda"
			sys.stdout.flush()
			self.update_lambda(doc_index)
			print "     Updating Nu"
			sys.stdout.flush()
			self.update_nu_squared(doc_index)
			print "     Updating Phi"
			sys.stdout.flush()
			self.update_phi(doc_index)
			after_bound = self.bound(doc_index, self.lambdas, self.nus_squared)
			print "     Bound comparison", before_bound, after_bound
			sys.stdout.flush()
			if abs((before_bound - after_bound)/before_bound) < self.convergence_cutoff:
				before_bound = after_bound
				break
			before_bound = after_bound
		return before_bound
	def EM(self):
		for it in xrange(self.max_iters):
			beta_estimated = np.zeros((self.num_topics, self.vocab_size))
			lambda_sum = np.zeros(self.num_topics)
			sigma_sum = np.zeros((self.num_topics, self.num_topics))
			nus_squared_d = []
			lambda_d = []


			# E step
			for doc_index in xrange(self.num_docs):
				print "E step for doc", doc_index
				sys.stdout.flush()
				self.reset_variational_parameters()
				self.update_variational_parameters(doc_index)

				# This multiplies each row (word) in phi, by the number of times that word appears in the doc
				# See http://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array for broadcasting description
				sys.stdout.flush()
				phi_weighted = np.multiply(self.phi, self.counts[doc_index][:, np.newaxis])
				# Add these to the beta_estimated variable. This will be useful info for the M-step.
				beta_estimated += phi_weighted.T
				self.saved_lambdas[doc_index] = self.lambdas
				lambda_d.append(self.lambdas)
				nus_squared_d.append(self.nus_squared)
			print lambda_d

			# M-Step
			# Normalize the betas
			for row in xrange(self.num_topics):
				beta_estimated[row] = beta_estimated[row]/np.sum(beta_estimated[row])

			self.beta = beta_estimated
			self.mu = sum(lambda_d)/len(lambda_d)

			for d in xrange(self.num_docs):
				shifted = lambda_d[d] - self.mu
				sigma_sum += np.diag(nus_squared_d[d]) + np.outer(shifted, shifted)
			sigma_sum /= self.num_docs
			self.sigma = sigma_sum
			self.sigma_inv = inv(self.sigma)



if __name__ == "__main__":

	ctm = CTM(4, 3, 3, np.array([[150, 12, 3], [17, 5, 1], [22, 2, 3], [1, 2, 3]]), 50, .001)
	ctm.EM()
	print ctm.sigma
	print ctm.beta

	x = np.array([1,2,3])
	y = np.array([2,4,6])
	ctm.update_nu_squared(2)

	a = np.ones((3, 3))



