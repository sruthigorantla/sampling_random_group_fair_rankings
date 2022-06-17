import numpy as np
from numpy.random import default_rng
import pandas as pd
import argparse

import copy 
import math

from itertools import product
from collections import defaultdict, Counter

from lattice_point_sampler import LatticePointSampler



class RandomWalkRanker(object):
	"""
	Helps create RandomWalkRanker object with dataset (or input) represented in terms of 
	(1) intra-group rankings,
	(2) fairness parameter and 
	(3) length of the ranking to be output
	"""

	def __init__(self, num_groups, intra_group_rankings, LB, UB, k):
		"""
		num_groups (int): the total number of disjoint groups
		intra_group_rankings (List[List[int]]) a list of list of intra group rankings represented by identifier of the items within the group
		LB (List[int]): list of integer lower bound on the number of top k ranks to be allocated to groups
		UB (List[int]): list of integer upper bound on the number of top k ranks to be allocated to groups
		k (int): length of the ranking to be output
		"""

		self.k = k
		self.num_groups = num_groups
		self.intra_group_rankings = intra_group_rankings

		self.LB = LB
		self.UB = UB
		self.sampler = LatticePointSampler(self.num_groups, self.LB, self.UB, self.k)



	def construct_ranking(self, answer):
		'''
		answer consists of group assignments,like [1,1,2,1,2,2,3] where the number represents group. 
		In this function we assign appropriate items from the intra-group ranking
		'''
		final_ranking = []

		intra_group_rankings = copy.deepcopy(self.intra_group_rankings)

		for item in answer:
			final_ranking.append(intra_group_rankings[item].pop(0))

		return final_ranking


	def sample_ranking(self, num_samples):

		# (1)
		# sample points from the sampler; 
		# this gives a uniform random group representation, x in the paper.
		all_sampled_points = np.array(self.sampler.sample(num_samples), dtype=int)
		

		assert len(all_sampled_points) == num_samples


		final_rankings = []
		for sampled_point in all_sampled_points:
			assert sum(sampled_point) == self.k, f"Total sum of sampled point is {sum(sampled_point)} instead of {self.k}"
			
			# (2)
			# Get a random permutation of groups for the ranking;
			# this gives a uniform random group assignment y given x. Same notation as used in the paper.
			permutation = []
			for group in range(self.num_groups):
				permutation += [group]*sampled_point[group]

			np.random.shuffle(permutation)

			# (3)
			# finally assing items in to the ranks allocated to the groups.
			final_rankings.append(self.construct_ranking(permutation))

		return final_rankings
		


if __name__ == "__main__":
	intra_group_rankings = [[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19]]
	num_groups = len(intra_group_rankings)
	LB = [1,1]
	UB = [9,9]
	k = 10
	random_walk_ranker = RandomWalkRanker(num_groups, intra_group_rankings, LB, UB, k)
	random_rankings = random_walk_ranker.sample_ranking(num_samples = 3)
	print("Sampled_rankings: ", random_rankings)