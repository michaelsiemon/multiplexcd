multiplexcd
===========

multiplexcd provides functions to perform multiplex community detection.

Installation
------------

This module relies on igraph's python extension as well as numpy and scipy. To install, run::

	pip install multiplexcd

Test
----

Test that the module is running correctly by running::

	python test.py

Usage
-----

Once the package has been installed with the necessary dependencies, the module performs three basic tasks. The core purpose is to generate a list of community ids for each vertex in each network::

    g = igraph.Graph()
    h = igraph.Graph(directed=True)

    g_vertex_names = ['a', 'b', 'c', 'd', 'e']
    h_vertex_names = ['a', 'b', 'c', 'd', 'e', 'f']

    g_edges = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('a', 'd'),
               ('d', 'e')]
    h_edges = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'a'),
               ('c', 'b'), ('d', 'e'), ('d', 'f'), ('e', 'f'),
               ('e', 'd'), ('f', 'd'), ('f', 'e')]

    g.add_vertices(g_vertex_names)
    h.add_vertices(h_vertex_names)

    g.add_edges(g_edges)
    h.add_edges(h_edges)

    omega = 1.0
    net_list = [g, h]
    net_types = ['s', 'd']

    community_ids = multiplexcd.multiplex_leading_eigenvector(
                        net_list, omega, net_types)

where the omega and net_types parameters define the specific form of the multiplex modularity matrix, which can be generated directly as shown below.::

    g.vs['memb'] = [0, 0, 0, 1, 1]
    h.vs['memb'] = [0, 0, 0, 1, 1, 1]

    community_ids = g.vs['memb'] + h.vs['memb']

    B, mu = multiplexcd.get_modularity_matrix(
                        net_list, omega, net_types)

This matrix may be used to get the multiplex modularity score for an arbitrary list of community ids.::

    Q = multiplexcd.multiplex_modularity(B, mu, community_ids)

Finally, the module provides a function to transform a list of distinct networks into a single, multislice representation with manually specified links between the same node in multiple networks.::

	multislice = multiplexcd.make_multislice_graph([g, h], omega)

This is especially useful for identifying connected components in the multiplex network at the cost of converting every graph into a single-mode, undirected network.

Author
------
Michael Siemon

License
-------
This project is licensed under the MIT License - see the LICENSE.md file for details.