import multiplexcd

from numpy.random import uniform
from random import randint, choice
from copy import copy

def random_biplex_graphs(n, e, k, mu):
	"""
	Creates a two random graphs with shared modular structure for testing.

	Parameters
	----------
	n:						int specifying the number of nodes
	e:						int specifying the number of edges
	k:						int specifying the number of communities
	mu:						float specifying the probability of 
							intra-community edges

	Returns g, h two igraph Graph instances with shared communities.
	"""
	g = ig.Graph(n = n)
	h = ig.Graph(n = n)

	g['resolution'] = 1.
	h['resolution'] = 1.

	g.vs['name'] = range(n)
	h.vs['name'] = range(n)

	n_k = n / k
	membership = [i/n_k for i in range(n)]

	g_ingroup = uniform(size = e)
	h_ingroup = uniform(size = e)

	g_edges = {i: [] for i in range(n)}
	h_edges = {i: [] for i in range(n)}

	for m in range(e):

		if g_ingroup[m] < mu:

			i = randint(0, n-1)
			c = membership[i]

			choices = [j for j in range(c*n_k, (c+1)*n_k) \
					if j not in g_edges[i] and j!=i]

			while len(choices)==0:
				i = randint(0, n-1)
				c = membership[i]

				choices = [j for j in range(c*n_k, (c+1)*n_k) \
						if j not in g_edges[i] and j!=i]

			j = choice(choices)

		else:
			i = randint(0, n-1)
			c = membership[i]
			choice_set = range(0, c * n_k) + range((c+1) * n_k, n)
			choices = [j for j in choice_set if j not in g_edges[i] and j!=i]
			j = choice(choices)

		g.add_edge(i, j)
		g_edges[i].append(j)
		g_edges[j].append(i)

		if h_ingroup[m] < mu:

			i = randint(0, n-1)
			c = membership[i]

			choices = [j for j in range(c*n_k, (c+1)*n_k) \
					if j not in h_edges[i] and j!=i]

			while len(choices)==0:
				i = randint(0, n-1)
				c = membership[i]

				choices = [j for j in range(c*n_k, (c+1)*n_k) \
						if j not in h_edges[i] and j!=i]

			j = choice(choices)

		else:
			i = randint(0, n-1)
			c = membership[i]
			choice_set = range(0, c * n_k) + range((c+1) * n_k, n)
			choices = [j for j in choice_set if j not in h_edges[i] and j!=i]
			j = choice(choices)

		h.add_edge(i, j)
		h_edges[i].append(j)
		h_edges[j].append(i)
	
	int2shape = {0:'circle', 1:'rect', 2:'diamond', 3:'triangle', 4:'down-triangle'}
	for i in range(k):
		if i not in int2shape:
			int2shape[i] = int2shape[i%5]
	g.vs['shape'] = [int2shape[m] for m in membership]
	h.vs['shape'] = [int2shape[m] for m in membership]

	return g, h

def test(n=100, e=200, k=5, mu=0.5, w=1.0, return_graphs=False):

	g, h = random_biplex_graphs(n, e, k, mu)
	u = multiplexcd.multiplex_leading_eigenvector([g, h], w, ['s', 's'])

	print 'Test successful'

	if return_graphs:
		return g, h

if __name__=='__main__':
	test()