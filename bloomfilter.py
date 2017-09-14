#!/usr/bin/env python3
'''A reference implementation of Bloom filter-based Iris-Code indexing.'''

__author__ = "Pawel Drozdowski"
__copyright__ = "Copyright (C) 2017 Hochschule Darmstadt"
__license__ = "License Agreement provided by Hochschule Darmstadt(https://github.com/dasec/bloom-filter-iris-indexing/blob/master/hda-license.pdf)"
__version__ = "1.0"

import argparse
import copy
import math
import operator
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, List, Set
import numpy as np

parser = argparse.ArgumentParser(description='Bloom filter-based Iris-Code indexing.')
parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
required = parser.add_argument_group('required named arguments')
required.add_argument('-d', '--directory', action='store', type=Path, required=True, help='directory where the binary templates are stored')
required.add_argument('-n', '--enrolled', action='store', type=int, required=True, help='number of enrolled subjects')
required.add_argument('-bh', '--height', action='store', type=int, required=True, help='filter block height')
required.add_argument('-bw', '--width', action='store', type=int, required=True, help='fitler block width')
required.add_argument('-T', '--constructed', action='store', type=int, required=True, help='number of trees constructed')
required.add_argument('-t', '--traversed', action='store', type=int, required=True, help='number of trees traversed')
args = parser.parse_args()

required_python_version = (3, 5)

if (sys.version_info.major, sys.version_info.minor) < required_python_version:
	sys.exit("Python {}.{} or newer is required to run this program".format(*required_python_version))

allowed_bf_heights = frozenset(range(8, 13))
allowed_bf_widths = frozenset({8, 16, 32, 64})

class BloomTemplate(object):
	'''Represents a Bloom Filter template or a Bloom Filter tree node'''

	def __init__(self, bloom_filter_sets: List[Set[int]], source: List[Tuple[str, str, str, str]]):
		self.bloom_filter_sets = bloom_filter_sets
		self.source = source

	def compare(self, other) -> float:
		'''Measures dissimilarity between two BloomTemplates'''
		return sum(len(s1 ^ s2) / (len(s1) + len(s2)) for s1, s2 in zip(self.bloom_filter_sets, other.bloom_filter_sets)) / len(self)

	def __add__(self, other):
		'''Merge two BloomTemplates by ORing their bloom filter sets'''
		return BloomTemplate([s1 | s2 for s1, s2 in zip(self.bloom_filter_sets, other.bloom_filter_sets)], self.source + [s for s in other.source if s not in self.source])

	def __iadd__(self, other):
		'''Add (OR) another template to self in-place'''
		self.bloom_filter_sets = [s1 | s2 for s1, s2 in zip(self.bloom_filter_sets, other.bloom_filter_sets)]
		self.source += (s for s in other.source if s not in self.source)
		return self

	def __len__(self) -> int:
		'''Number of bloom filters in the template'''
		return len(self.bloom_filter_sets)

	def __getitem__(self, key: int) -> Set[int]:
		'''Convenience access for individual bloom filters in the template'''
		return self.bloom_filter_sets[key]

	def __repr__(self) -> str:
		return "Bloom filter template of {}".format(self.source)

	# Convenience functions for template source comparison
	def is_same_subject(self, other) -> bool:
		return len(self.source) == len(other.source) and all(s_item[0] == o_item[0] for s_item, o_item in zip(self.source, other.source))

	def is_same_image(self, other) -> bool:
		return len(self.source) == len(other.source) and all(s_item[1] == o_item[1] for s_item, o_item in zip(self.source, other.source))

	def is_same_side(self, other) -> bool:
		return len(self.source) == len(other.source) and all(s_item[2] == o_item[2] for s_item, o_item in zip(self.source, other.source))

	def is_same_dataset(self, other) -> bool:
		return len(self.source) == len(other.source) and all(s_item[3] == o_item[3] for s_item, o_item in zip(self.source, other.source))

	def is_same_genuine(self, other) -> bool:
		return len(self.source) == len(other.source) and self.is_same_subject(other) and self.is_same_side(other) and self.is_same_dataset(other)

	def is_same_source(self, other) -> bool:
		return len(self.source) == len(other.source) and all(s_item == o_item for s_item, o_item in zip(self.source, other.source))

	def is_multi_source(self) -> bool:
		return len(self.source) > 1

	@classmethod
	def from_binary_template(cls, binary_template: List[List[int]], height: int, width: int, source: List[Tuple[str, str, str, str]]):
		'''Creates a BloomTemplate with specified block size from an iris code represented as a 2-dimensional (row x column) array of 0's and 1's. The source is a list of tuples following format: [(subject, image_number, side, dataset), ...]'''
		if height not in allowed_bf_heights or width not in allowed_bf_widths:
			raise ValueError("Invalid block size: ({}, {})".format(height, width))
		binary_template = np.array(binary_template)
		bf_sets = []
		bf_real = set()
		bf_imaginary = set()
		for column_number, column in enumerate(binary_template.T):
			real_part = ''.join(map(str, column[:height]))
			im_part_start = 10 if height <= 10 else len(binary_template) - height
			im_part_end = im_part_start + height
			imaginary_part = ''.join(map(str, column[im_part_start:im_part_end]))
			bf_value_real = int(real_part, 2)
			bf_value_imaginary = int(imaginary_part, 2)
			bf_real.add(bf_value_real)
			bf_imaginary.add(bf_value_imaginary)
			if column_number != 0 and (column_number + 1) % width == 0:
				bf_sets.append(bf_real)
				bf_sets.append(bf_imaginary)
				bf_real = set()
				bf_imaginary = set()
		return BloomTemplate(bf_sets, source)

BF_TREE = List[BloomTemplate]
class BloomTreeDb(object):
	'''Represents a database of BloomTemplate trees'''

	def __init__(self, enrolled: List[BloomTemplate], trees_constructed: int):
		def is_power_of2(number: int) -> bool:
			'''Check if a number is a power of 2.'''
			return number > 0 and (number & (number - 1)) == 0
		if not is_power_of2(len(enrolled)) or not is_power_of2(trees_constructed):
			raise ValueError("Number of subjects ({}) and trees ({}) must both be a power of 2".format(len(enrolled), trees_constructed))
		self.enrolled = enrolled
		self.trees_constructed = trees_constructed
		self.trees = self._build()

	def search(self, probe: BloomTemplate, trees_traversed: int) -> Tuple[float, BloomTemplate]:
		'''Perform a search for a template matching the probe in the database.'''
		def find_promising_trees(probe: BloomTemplate, trees_traversed: int) -> List[BF_TREE]:
			'''Preselection step - most promising trees are found based on the scores between the tree roots and the probe'''
			if self.trees_constructed == trees_traversed:
				return self.trees
			else:
				root_scores = [(tree[0].compare(probe), index) for index, tree in enumerate(self.trees)]
				root_scores.sort(key=operator.itemgetter(0))
				promising_tree_indexes = map(operator.itemgetter(1), root_scores[:trees_traversed])
				return [self.trees[index] for index in promising_tree_indexes]
		def traverse(trees: List[BF_TREE], probe: BloomTemplate) -> Tuple[float, BloomTemplate]:
			'''Traverse the selected trees to find the node corresponding to a best score'''
			best_score, best_match_node = 1.0, None
			for _, tree in enumerate(trees):
				step = 0
				score = 1.0
				for _ in range(int(math.log(len(self.enrolled), 2)) - int(math.log(self.trees_constructed, 2))):
					left_child_index, right_child_index = BloomTreeDb.get_node_children_indices(step)
					ds_left = tree[left_child_index].compare(probe)
					ds_right = tree[right_child_index].compare(probe)
					step, score = (left_child_index, ds_left) if ds_left < ds_right else (right_child_index, ds_right)
				score, match_node = score, tree[step]
				if score <= best_score:
					best_score = score
					best_match_node = match_node
			return best_score, best_match_node

		if trees_traversed < 1 or trees_traversed > self.trees_constructed:
			raise ValueError("Invalid number of trees to traverse:", trees_traversed)
		promising_trees = find_promising_trees(probe, trees_traversed)
		return traverse(promising_trees, probe)

	def _build(self) -> List[BF_TREE]:
		'''Constructs the BloomTemplate trees using the parameters the db has been initiated with'''
		def construct_bf_tree(enrolled_part: List[BloomTemplate]) -> BF_TREE:
			'''Constructs a single BloomTemplate tree'''
			bf_tree = []
			for index in range(len(enrolled_part)-1):
				node_level = BloomTreeDb.get_node_level(index)
				start_index = int(len(enrolled_part) / (1 << node_level) * ((index + 1) % (1 << node_level)))
				end_index = int(len(enrolled_part) / (1 << node_level) * ((index + 1) % (1 << node_level)) + len(enrolled_part) / (1 << node_level))
				node = copy.deepcopy(enrolled_part[start_index])
				for i in range(start_index, end_index):
					node += enrolled_part[i]
				bf_tree.append(node)
			bf_tree += enrolled_part
			return bf_tree
		trees = []
		i = 0
		while i != len(self.enrolled):
			i_old = i
			i += int(len(self.enrolled) / self.trees_constructed)
			bf_tree = construct_bf_tree(self.enrolled[i_old:i])
			assert len(bf_tree) == int(len(self.enrolled) / self.trees_constructed) * 2 - 1
			trees.append(bf_tree)
		assert len(trees) == self.trees_constructed
		return trees

	def __repr__(self) -> str:
		return "<BloomTreeDb object containing {} subjects in {} trees>".format(len(self.enrolled), self.trees_constructed)

	'''Convenience methods for tree indexing'''
	@staticmethod
	def get_node_children_indices(index: int) -> Tuple[int, int]:
		'''Compute indices of node children based on its index.'''
		return 2 * index + 1, 2 * (index + 1)

	@staticmethod
	def get_node_level(index: int) -> int:
		'''Compute the level of a node in a tree based on its index.'''
		return int(math.floor(math.log(index + 1, 2)))

def load_binary_template(path: Path) -> List[List[int]]:
	'''Reads a text file into an iris code matrix'''
	with path.open("r") as f:
		return [list(map(int, list(line.rstrip()))) for line in f.readlines()]

def extract_source_data(filename: str) -> List[Tuple[str, str, str, str]]:
	'''This function parses the template filename (path.stem) and extract the subject, image number, image side and dataset and return it as list (this is necessary later on) with one tuple element (Subject, Image, Side, Dataset).
	e.g. if the filename is "S1001L01.jpg" from Casia-Interval dataset, then the return value should be: [(1001, 01, L, Interval)] or similar, as long as the convention is consistent.
	'''
	raise NotImplementedError("Implement me!")

def split_dataset(templates: List[BloomTemplate], num_enrolled: int) -> Tuple[List[BloomTemplate], List[BloomTemplate], List[BloomTemplate]]:
	'''This function splits the full template list into disjoint lists of enrolled, genuine and impostor templates'''
	enrolled, genuine, impostor = [], [], []
	raise NotImplementedError("Implement me!")
	return enrolled, genuine, impostor

if __name__ == "__main__":
	# Data preparation
	start = timer()
	binary_templates = [(load_binary_template(f), extract_source_data(f.stem)) for f in args.directory.iterdir() if f.is_file() and f.match('*.txt')] # see file example_binary_template.txt for required format
	bloom_templates = [BloomTemplate.from_binary_template(template, args.height, args.width, source) for template, source in binary_templates]
	enrolled_templates, genuine_templates, impostor_templates = split_dataset(bloom_templates, args.enrolled)
	db = BloomTreeDb(enrolled_templates, args.constructed)
	end = timer()
	print("Total data preparation time: %02d:%02d" % divmod(end - start, 60))

	# Lookup
	start = timer()
	results_genuine = [db.search(genuine_template, args.traversed) for genuine_template in genuine_templates] # List[Tuple[float, BloomTemplate]]
	results_impostor = [db.search(impostor_template, args.traversed) for impostor_template in impostor_templates] # List[Tuple[float, BloomTemplate]]
	genuine_scores = [result[0] for result in results_genuine] # List[float]
	impostor_scores = [result[0] for result in results_impostor] # List[float]
	genuine_matches = [result[1] for result in results_genuine] # List[BloomTemplate]
	end = timer()
	print("Total lookup time: %02d:%02d" % divmod(end - start, 60))

	# Results
	print("Experiment configuration: {} enrolled, {} trees, {} traversed trees, {} block height, {} block width".format(len(enrolled_templates), args.constructed, args.traversed, args.height, args.width))
	print("Genuine distribution: {} scores, min/max {:.4f}/{:.4f}, mean {:.4f} +/- {:.4f}".format(len(genuine_scores), min(genuine_scores), max(genuine_scores), np.mean(genuine_scores), np.std(genuine_scores)))
	print("Impostor distribution: {} scores, min/max {:.4f}/{:.4f}, mean {:.4f} +/- {:.4f}".format(len(impostor_scores), min(impostor_scores), max(impostor_scores), np.mean(impostor_scores), np.std(impostor_scores)))
	print("Fraction of genuine attempts with correct leaf reached: {:.4f}".format([probe.is_same_genuine(result) for probe, result in zip(genuine_templates, genuine_matches)].count(True) / len(genuine_templates)))
