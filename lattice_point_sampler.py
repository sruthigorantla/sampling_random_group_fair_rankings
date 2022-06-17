import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()
s = eng.genpath('./PolytopeSamplerMatlab-master')
eng.addpath(s, nargout=0)
import matlab


class LatticePointSampler:
	def __init__(self,num_groups,lb,ub,k):
		self.num_groups = num_groups
		self.k = k
		self.lb = lb
		self.ub = ub

	def sample(self, num_samples):
		done = False
		sample_count = 0
		sampled_points = []
		while sample_count < num_samples:
			self.delta = self.compute_delta()
			self.x_star = self.get_x_star()
			lb_new = ((self.lb - self.x_star)*(1+(np.sqrt(self.num_groups)/self.delta))).reshape(-1,1).tolist()
			ub_new = ((self.ub - self.x_star)*(1+(np.sqrt(self.num_groups)/self.delta))).reshape(-1,1).tolist()
			z = np.transpose(np.asarray(eng.sampling_from_simplex(self.num_groups,self.k,matlab.double(lb_new),matlab.double(ub_new),num_samples-sample_count)))
			for point in z:
				x = self.round(point)
				if self.inN(x):
					sampled_points.append(x)
					sample_count += 1
		return sampled_points[:num_samples]

	def compute_delta(self):
		minimum = np.inf
		for j in range(int(self.num_groups)):
			minimum = min(minimum, 0.5*(self.ub[j]-self.lb[j]) - 1)
		minimum = min(minimum, (self.k - np.sum(self.lb))/self.num_groups - 1)
		minimum = min(minimum, (np.sum(self.ub) - self.k)/self.num_groups - 1)
		return minimum

	def get_x_star(self):
		x = np.zeros(int(self.num_groups))
		x = self.lb + np.ceil(self.delta)
		while np.sum(x) < self.k:
			j = np.argmax(self.ub - np.ceil(self.delta) - x)
			x[j] = min( self.k - np.sum(x) + x[j], self.ub[j] - np.ceil(self.delta))
		return x
	
	def round(self,z):
		y = z + self.x_star
		sorted_indices = np.argsort(y-np.floor(y))[::-1][:len(z)]

		for j in range(int(self.k - np.sum(np.floor(y)))):
			y[sorted_indices[j]] = np.floor(y[sorted_indices[j]]) + 1
		j = int(self.k - np.sum(np.floor(y)))
		while j < self.num_groups:
			y[sorted_indices[j]] = np.floor(y[sorted_indices[j]]) 
			j += 1
		return y

	def inN(self,x):
		if np.all(np.less_equal(self.lb,x)) and np.all(np.greater_equal(self.ub,x)) and np.sum(x) == self.k:
			return True

	def plot(self):
		plt.scatter(s[0], s[2])
		plt.savefig("test.png")