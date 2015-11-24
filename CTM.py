import numpy as np
from numpy.linalg import inv, det
from scipy.optimize import fmin_cg, minimize, check_grad


class CTM:
	# counts is an array of num_docs x vocab_size, where the element is the number of counts of that word.
	def __init__(self, num_docs, num_topics, vocab_size, counts, max_iters, convergence_cutoff):
		self.K = num_topics
		self.num_docs = num_docs
		self.mu = np.zeros(num_topics)
		self.sigma = np.diag(np.ones(num_topics))
		self.sigma_inv = inv(self.sigma)
		#self.betaa = np.ones((num_topics, vocab_size))
		self.beta =  np.random.uniform(0, 1, (num_topics, vocab_size))
		self.lambdas = np.zeros(num_topics)
		self.nus_squared = np.ones(num_topics)
		self.phi = 1.0/num_topics*np.ones((vocab_size, num_topics))
		self.update_zeta()

		
		self.counts = counts
		self.vocab_size = vocab_size
		self.max_iters = max_iters
		self.convergence_cutoff = convergence_cutoff

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
			'disp' : True,    # non-default value.
			'gtol' : 1e-5}
		res = minimize(lambda x: -obj(x), self.lambdas, jac=lambda x: -derivative(x), method='CG', options=opts)
		self.lambdas = res.x
		#print obj(res2.x)
		#print derivative(res2.x)
		#print res2.x
		#print res2
		#print self.lambdas
		#print obj([0,0]), obj([1, 0]), obj([-1, 0])
		#print derivative([0,0]), derivative([1, 0]), derivative([-1, 0])

		#for rep in range(100):
		#	print check_grad(lambda x: -obj(x), lambda y: -derivative(y), np.random.uniform(0, .0002, 2))
	
	def update_nu_squared(self, doc_index):
		def obj(nu_sq):
			#N = self.getWordCount(doc_index)
			#total = 0
			#total -= .5*(np.trace(np.dot(np.diag(nu_sq), self.sigma_inv)))
			#total += N*(-1./self.zeta*(np.sum(np.exp(self.lambdas + nu_sq/2))))
			#total += .5*(np.sum(np.log(nu_sq)))
			#print total
			#return total
			return self.bound(doc_index, self.lambdas, nu_sq)
		def derivative(nu_sq):
			N = self.getWordCount(doc_index)
			grad = np.zeros(self.K)
			grad += -.5*np.diag(self.sigma_inv)
			grad += -N/(2.*self.zeta)*np.exp(self.lambdas + nu_sq/2) 
			grad += .5/nu_sq
			return grad
			#result = np.zeros(self.K)
			#for i in xrange(self.K):
			#	result[i] = -.5*self.sigma_inv[i,i]
			#	result[i] -= N/(2*self.zeta) * np.exp(self.lambdas[i] + .5*nu_sq[i])
			#	result[i] += 1/(2*nu_sq[i])
			#return result
		bounds = [(0, None) for i in range(self.K)]
		res = minimize(lambda x: -obj(x), self.nus_squared, jac=lambda x: -derivative(x), method='L-BFGS-B', bounds=bounds)
		self.nus_squared = res.x
		#for rep in range(100):
		#	print check_grad(lambda x: -obj(x), lambda y: -derivative(y), np.random.uniform(1., 10.0002, 2))

	# This is based on equations 8-12 from CTM paper
	def bound(self, doc_index, lamdas, nu_squared_vals):
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
		before_bound = self.bound(doc_index, self.lambdas, self.nu_squared_vals)
		for it in xrange(self.max_iters):
			self.update_zeta()
			self.update_lambda(doc_index)
			self.update_nu_squared(doc_index)
			self.update_phi(doc_index)
			after_bound = self.bound(doc_index, self.lambdas, self.nu_squared_vals)
			if abs((before_bound - after_bound)/before_bound) < self.convergence_cutoff:
				before_bound = after_bound
				break
			before_bound = after_bound
		return before_bound
ctm = CTM(4, 2, 3, np.array([[1, 2, 3], [2, 1, 3], [2, 2, 2], [1, 4, 5]]), 50, .001)
print ctm.K
print ctm.lambdas
print ctm.nus_squared
print ctm.zeta
print ctm.phi

x = np.array([1,2,3])
y = np.array([2,4,6])
print 1./x
#print np.sum(ctm.phi, axis=0)
#ctm.update_lambda(2)
ctm.update_nu_squared(2)

print np.dot(x, x.transpose())



