"""Functions to perform multiplex community detection.

This module provides functions to simultaneously analyze the community
structure common to a list of igraph.Graph instances. It works by performing
spectral partitions of a multiplex modularity matrix, which is formed by
a block-diagonal arrangement of modularity matrices from each graph and
sparse off-diagonal entries that serve as the links between different
networks in the multiplex structure. It refines these partitions with a
Kernighan-Lin algorithm and by ensuring the connectivity of each community.
Networks may be symmetric, directed, or bipartite. Weighted and unweighted
networks are both supported. 

This module relies on the igraph_ package for python. With igraph
installed, an analysis of one symmetric and one directed graph might proceed 
as follows::

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

To measure the multiplex modularity of an arbitrary community id vector::
    
    g.vs['memb'] = [0, 0, 0, 1, 1]
    h.vs['memb'] = [0, 0, 0, 1, 1, 1]

    community_ids = g.vs['memb'] + h.vs['memb']

    B, mu = multiplexcd.get_modularity_matrix(
                        net_list, omega, net_types)

    Q = multiplexcd.multiplex_modularity(B, mu, community_ids)

The sets of vertices in each network may differ.

For bipartite networks, the vertices must be sorted by type so that the
block-diagonal portions of the network's adjacency matrix are all 0.

To alter the size of the communities returned by the algorithm, each Graph
instance may be assigned its own 'resolution' attribute as follows::
    
    gamma_g, gamma_h = 1.0, 1.5
    g['resolution'] = gamma_g
    h['resolution'] = gamma_h

where gamma_g and gamma_h modify the penalty for grouping unconnected vertices 
in the same community. Higher values yield smaller communities.
"""
__version__ = '1.0'
__author__ = 'Michael Siemon'

import itertools

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sLA

import igraph as ig


def multiplex_modularity(B, mu, membership):
    """Calculates a multiplex modularity score.

    Calculates the modularity from a given modularity matrix and membership
    vector.

    Args:
        B (scipy.sparse.csr_matrix): An n by n sparse modularity matrix where
            n is the number of vertices across all networks.

        mu (float): The total multislice strength (see Mucha et al. 2010).

        membership (list): A vector of community ids of length n.

    Returns:
        float. The modularity value.
    """

    c2comemb = {}
    clist = list(set(membership))
    clist.sort()

    for c in clist:
        c2comemb[c] = np.matrix([1 if c==m else 0 for m in membership])

    Q = np.matrix([B.getrow(i).dot(c2comemb[m].T)[0,0]
                   for i, m in enumerate(membership)])

    return Q.sum() / (2 * mu)

# # # # # # # # # # # # #
#  Main functions
# # # # # # # # # # # # #


def multiplex_leading_eigenvector(net_list, w, net_types, weight='weight',
                                  id_attr='name', verbose=False):
    """Partitions vertices from multiple networks into communities.

    Detect communities using a multiplex adaptation of Newman's (2006) leading
    eigenvector method.

    Args:
        net_list (list): Contains igraph.Graph instances. Each graph may have a
            'resolution' attribute, which defaults to 1.0 if not specified. The
            multislice network jointly defined by the graphs and the w
            parameters should have exactly one component.

        w: Float or dictionary of the form:

            {(i, j): interslice_weight for i, j in
            itertools.permutation(range(len(net_list)), 2)}

        net_types (list): Contains strings specifying the modularity equation
            to use for each Graph instance in net_list. Can include::

                's' -- symmetric
                'd' -- directed
                'b' -- bipartite
             
        weight (str): Attribute specifying edge weight. Defaults to 'weight'.
            Use None to specify using edge count.
        id_attr (str): Attribute for matching vertex identities across slices.
            Defaults to 'name'.

    Returns:
        list. Community ids of each vertex.
    """
    B, mu = get_modularity_matrix(net_list, w, net_types,
                                  weight=weight, id_attr=id_attr)

    try:
        # Perform spectral split
        eigvals, eigvectors = sLA.eigsh(B, k=1)

        if eigvectors is None:
            return []

        u = eigvectors.T.tolist()[0]

        u = [0 if v < 0 else 1 for v in u]

        membership = copy(u)

        # Kernighan-Lin refinement algorithm
        while membership is not False:
            membership = KL_refinement(B, membership, mu)
            if membership is not False:
                u = membership

        # Ensure connectivity of two communities
        g = make_multislice_graph(net_list, w)
        g['subgraph_membership'] = u
        g.vs['subgraph_memb'] = u
        u = _ensure_connectivity(g)

        if verbose:
            Q = multiplex_modularity(B, mu, u)
            print 'Initial_split:', Q, sum(u)

        _recursive_split(B, g, u, 0, 1, mu, verbose=verbose)

    except sLA.ArpackNoConvergence:
        u = [0 for i in range(B.shape[0])]
        print "Convergence Error: No partition made.", B.shape

    # Save results as igraph Graph attributes
    start_idx = 0
    for g in net_list:
        g['membership'] = u[start_idx: start_idx + len(g.vs)]
        g['clustering'] = ig.VertexClustering(g, g['membership'])
        start_idx += len(g.vs)

    print len(set(u))
    print '-----------------'

    return u


def _recursive_split(B, g, membership, m1, m2, mu, verbose=False):
    """Provides a recursive, branching wrapper around _attempt_split function.

    Implements the recursive bifurcation of each community found in a previous
    split until no modularity increase can be found. Alters the existing
    membership list.

    Args:
        B (scipy.sparse.csr_matrix): An n by n sparse modularity matrix where
            n is the number of vertices across all networks.
        g (igraph.Graph): Holds the multislice representation of
            the networks used to form B
        membership (list): Community ids
        m1 (int): The id of the first community to target for split
        m2 (int): The id of the second community to target for split
        mu (float): The total multislice strength (see Mucha et al. 2010).
        verbose (bool): Whether to print the results of intermediate
            stages of the algorithm.
    """

    # Attempt to generate a new community value from a split of community m1
    new_c = _attempt_split(B, g, membership, m1, mu, verbose=verbose)

    if new_c is not False:
        # Continue spliting
        _recursive_split(B, g, membership, m1, new_c, mu, verbose=verbose)

    # Repeat for m2
    new_c = _attempt_split(B, g, membership, m2, mu, verbose=verbose)

    if new_c is not False:
        _recursive_split(B, g, membership, m2, new_c, mu, verbose=verbose)


def _attempt_split(B, g, membership, target, mu, verbose=False):
    """Tries to find a modularity-increasing split of the target community.

    Args:

        B (scipy.sparse.csr_matrix): An n by n sparse modularity matrix where
            n is the number of vertices across all networks.
        g (igraph.Graph): Holds the multislice representation of the networks
            used to form B.
        membership (list): Community ids.
        target (int): Id of community to target for split.
        mu (float): The total multislice strength (see Mucha et al. 2010).
        verbose (bool): Whether to print the results of intermediate stages
            of the algorithm.

    Returns:
        int.  Id of new community if one is found, else False
    """
    # Select all vertices in target community.
    idxs = [i for i, m in enumerate(membership) if m == target]

    # Submatrix of community to be split.
    B_g = B[idxs, :].tocsc()[:, idxs].tocsr()

    # Subtrack row sums from diagonal of B_g as per Newman (2006).
    B_g -= sparse.dia_matrix((B_g.sum(axis = 1).T, np.array([0])),
                              shape=(len(idxs), len(idxs)))

    # Spectral partition.
    try:
        eigvals, eigvectors = sLA.eigsh(B_g, k=1)

        l = eigvals[0]
        u = eigvectors.T.tolist()[0]
        u = [0 if v < 0 else 1 for v in u]

        membershipkl = copy(u)

        s = np.matrix([-1 if v==0 else 1 for v in u])

        delta_Q = (1. / (4*mu)) * (s * B_g * s.T)[0,0]

        while membershipkl is not False:
            membershipkl = KL_refinement(B_g, membershipkl, mu)
            if membershipkl is not False:
                u = membershipkl

        # Create new membership attributes for use in _ensure_connectivity.
        g['subgraph_membership'] = [m + 2 for m in membership]
        g.vs['subgraph_memb'] = [m + 2 for m in membership]

        for i, idx in enumerate(idxs):
            g['subgraph_membership'][idx] = u[i]
            g.vs[idx]['subgraph_memb'] = u[i]

        subgraph_membership = _ensure_connectivity(g)
        for i, idx in enumerate(idxs):
            u[i] = subgraph_membership[idx]

    except sLA.ArpackNoConvergence:
        u = [0 for idx in idxs]
        if verbose:
            print "Convergence Error", B_g.shape

    except ValueError:
        u = [0 for idx in idxs]
        if verbose:
            print "Modularity Matrix too small", B_g.shape

    # Calculate modularity improvement.
    s = np.matrix([-1 if v == 0 else 1 for v in u])

    delta_Q = (1. / (4*mu)) * (s * B_g * s.T)[0,0]

    if delta_Q > 10**-15:

        # If split yields a modularity increase, save results.
        new_c = max(membership) + 1

        for i, idx in enumerate(idxs):
            m = u[i]
            if m==1:
                membership[idx] = new_c
        if verbose:
            print 'Old index:', target, 'New Index:', new_c, '+Q =', delta_Q,
            print sum([m == target for m in membership]),
            print sum([m == new_c for m in membership])

        return new_c

    else:

        if verbose:
            print 'Split rejected.', target, len(idxs)

        return False


# # # # # # # # # # # # #
#  Modularity Matrix
# # # # # # # # # # # # #

def get_modularity_matrix(net_list, w, net_types,
                          weight='weight', id_attr='name'):
    """Get a modularity matrix from a list of networks.

    Calculates the modularity matrix for a group of multiplex networks.
    Networks can be either weighted or unweighted and symmetric, directed, and
    bipartite. Bipartite graphs require that vertices are sorted by type, and
    thus that all edges are observed on the off-diagonal blocks of the
    adjacency matrix.

    Args:
        net_list (list): Contains igraph.Graph instances. Each graph may have a
            'resolution' attribute, which defaults to 1.0 if not specified. The
            multislice network jointly defined by the graphs and the w
            parameters should have exactly one component.

        w: Float or dictionary of the form:

            {(i, j): interslice_weight for i, j in
            itertools.permutation(range(len(net_list)), 2)}

        net_types (list): Contains strings specifying the modularity equation
            to use for each Graph instance in net_list. Can include::

                's' -- symmetric
                'd' -- directed
                'b' -- bipartite
             
        weight (str): Attribute specifying edge weight. Defaults to 'weight'.
            Use None to specify using edge count.
        id_attr (str): Attribute for matching vertex identities across slices.
            Defaults to 'name'.

    Returns:
        scipy.sparse.csr_matrix. A modularity matrix composed of block-diagonal
        modularity matrices specific to each network type and manually
        specified links across networks.
        float. A measure of multislice strength.
    """
    for net in net_list:
        if weight not in net.es.attribute_names():
            net.es['weight'] = 1.

    # Calculate intra-slice modularity.
    B, mu = _diag_modularity(net_list, net_types, weight=weight)
    B = B.tocsr()

    # Add inter-slice connections.
    B_offdiag, mu = _multislice_connections(net_list, w, mu, id_attr='name')
    B += B_offdiag

    # Make B symmetric.
    B += B.T
    B *= 0.5

    return B, mu


def _diag_modularity(net_list, net_types, weight='weight', id_attr='name'):
    """Creates the block-diagonal components of a modularity matrix.

    Calculates the intra-slice modularity matrices for a group of
    multiplex networks. Networks can be either weighted or unweighted and
    symmetric, directed, and bipartite. Bipartite graphs require that vertices
    are sorted by type, and thus that all edges are observed on the
    off-diagonal blocks of the adjacency matrix.

    Args:
        net_list (list): Contains igraph.Graph instances. Each graph may have a
            'resolution' attribute, which defaults to 1.0 if not specified. The
            multislice network jointly defined by the graphs and the w
            parameters should have exactly one component.

        net_types (list): Contains strings specifying the modularity equation
            to use for each Graph instance in net_list. Can include::

                's' -- symmetric
                'd' -- directed
                'b' -- bipartite

        weight (str): Attribute specifying edge weight. Defaults to 'weight'.
            Use None to specify using edge count.
  
        id_attr (str): Attribute for matching vertex identities across slices.
            Defaults to 'name'.

    Returns:
        scipy.sparse.block_diag. A modularity matrix composed of block-diagonal
        modularity matrices specific to each network type.
        float. A measure of intra-slice strength.
    """

    n = sum([len(g.vs) for g in net_list])

    mu = 0.

    B = []
    n = 0

    for s, g in enumerate(net_list):

        type_s = net_types[s]
        n_s = len(g.vs)

        if n_s == 0:
            continue

        try:
            gamma = g['resolution']
        except KeyError:
            gamma = 1.

        try:
            A_s = np.matrix(g.get_adjacency(attribute = weight).data)
        except ValueError:
            A_s = np.matrix(g.get_adjacency().data)

        if type_s == 's':  # Null model from Newman (2006)

            k = A_s.sum(axis = 1)
            m = A_s.sum() * .5

            A_s -= gamma*k*k.T / (2.*m)

        elif type_s == 'b':  # Null model from Barber (2006)

            k = A_s.sum(axis = 1)
            m = A_s.sum() * .5

            assert g.is_bipartite(), \
                  'Graph with net_type "b" is not bipartite.'

            n_total = len(g.vs)

            n_bottom = sum(g.vs['type'])
            n_top = n_total - n_bottom

            try:
                assert sum(A_s[:n_top, :n_top])[0,0] == 0, \
                      'Bipartite adjacency matrix not sorted by vertex type.'
            except TypeError:  # Ignore empty matrices
                pass

            P_s = np.matrix(np.zeros((n_total, n_total)))
            P_s_full = (gamma * k * k.T / m)
            P_s[n_top:, :n_top] = P_s_full[n_top:, :n_top]
            P_s[:n_top, n_top:] = P_s_full[:n_top, n_top:]

            A_s-=P_s
            P_s = None

        else:  # Null model from Leicht and Newman (2008)

            assert type_s == 'd', 'net_type must be either s, b, or d'

            k_in = A_s.sum(axis = 0)
            k_out = A_s.sum(axis = 1)
            m = float(A_s.sum())

            if m > 0:
                A_s -= gamma * k_out * k_in / m

        B.append(A_s)

        mu += m
        n += n_s

    return sparse.block_diag(B), mu


def _multislice_connections(net_list, w, mu, id_attr='name'):
    """Calculates the off-diagonal modularity for a group of networks.

    Args:
        net_list (list): Contains igraph.Graph instances. Each graph may have a
            'resolution' attribute, which defaults to 1.0 if not specified. The
            multislice network jointly defined by the graphs and the w
            parameters should have exactly one component.

        w: Float or dictionary of the form::

            {(i, j): interslice_weight for i, j in
            itertools.permutation(range(len(net_list)), 2)}


        net_types (list): Contains strings specifying the modularity equation
            to use for each Graph instance in net_list. Can include::

                's' -- symmetric
                'd' -- directed
                'b' -- bipartite

        weight (str): Attribute specifying edge weight. Defaults to 'weight'.
            Use None to specify using edge count.

        id_attr (str): Attribute for matching vertex identities across slices.
            Defaults to 'name'.

    Returns:
        scipy.sparse.csr_matrix. A modularity matrix composed of the manually
            specified links across networks.
        float. A measure of inter-slice strength.
    """
    # Populate a dictionary mapping vertex names to their index
    for g in net_list:
        g['name2id'] = {v[id_attr]: v.index for v in g.vs}

    data = []
    row_idxs = []
    col_idxs = []

    row_idx = 0
    ijs = []
    for s, g in enumerate(net_list):
        column_idx = 0

        for r, h in enumerate(net_list):
            if r <= s:
                column_idx += len(h.vs)
                continue

            try:
                C = w[(s, r)]
            except KeyError:
                try:
                    C = w[(r, s)]
                except KeyError:
                    C = 0.
            except TypeError:
                C = w

            if C:
                for n in g.vs['name']:
                    if n in h['name2id']:
                        i = g['name2id'][n] + row_idx
                        j = h['name2id'][n] + column_idx
                        assert i!=j;
                        data.append(C)
                        row_idxs.append(i)
                        col_idxs.append(j)

                        data.append(C)
                        row_idxs.append(j)
                        col_idxs.append(i)

                        mu += C

            column_idx += len(h.vs)

        row_idx += len(g.vs)

    # Add fake connection to bottom right corner to set matrix size.
    data.append(1)
    row_idxs.append(row_idx-1)
    col_idxs.append(row_idx-1)

    B = sparse.csr_matrix((data, (row_idxs, col_idxs)))

    # Delete place-holder ensuring B.shape == (row_idx, row_idx).
    B[row_idx-1, row_idx-1] = 0

    assert B.sum()%2 == 0

    return B, mu

# # # # # # # # # # # # #
#  Partition Refinment
# # # # # # # # # # # # #


def KL_refinement(B, membership, mu, verbose=False):
    """Improves a given two-way partition using the KL algorithm.

    Searches for higher-modularity partitions by switching each vertex once in
    the order of the change in modularity resulting from the move. For larger
    sets of networks with a total of over 10,000 vertices, the algorithm will
    cease searching for a better partition after 2000 failed attempts.

    Args:
        B (scipy.sparse.csr_matrix): An n by n sparse modularity matrix where
            n is the number of vertices across all networks.

        membership (list): A vector of community ids of length n.

        mu (float): The total multislice strength (see Mucha et al. 2010).

    Returns:
        Refined community membership list of length N if successful, otherwise
        the bool False
    """
    # Convert membership into two opposite vectors.
    m1 = np.matrix(membership)
    m0 = 1 - m1
    n = len(membership)

    # w is the sum of each row's inter-slice connections.
    w = B.sum(axis=1)

    # Q is a list of each row's contribution to the total modularity under
    # the input membership list.
    Q_1 = B.dot(m1.T)
    Q_0 = B.dot(m0.T)

    Q = np.matrix([Q_1[i,0] if membership[i] == 1 else Q_0[i,0]
                   for i in range(n)])

    start_q = Q.sum() / (2. * mu)

    # DQ is the change in moodularity produced by flipping each row's
    # community assignment.

    w = w.T + B.diagonal()
    DQ = w - 2*Q
    DQ = DQ.tolist()[0]

    # Make a set to keep track of switched vertices.
    moved = set()

    # Make objects to store values over iterations.
    mi1 = copy(m1)
    mi0 = 1 - mi1
    maxq = start_q
    maxqm = mi1.tolist()[0]
    q2m = {}

    # Set tolerance for number of modularity decreasing moves.
    if len(DQ) > 10000:
        max_fail = 2000
    else:
        max_fail = None

    consec_fail = 0

    # Move each vertex exactly once.
    qdata = []
    for k in range(n):
        consec_fail += 1

        # Get unmoved vertex with maximum change in modularity.
        maxdq = max(DQ)
        idx = DQ.index(maxdq)

        if idx in moved:
            indices = [i for i, q in enumerate(DQ)
                       if q == maxdq and i not in moved]
            idx = indices[0]

        moved.add(idx)

        mi1[0,idx] = 1 - mi1[0,idx]
        mi0[0,idx] = 1 - mi1[0,idx]

        new_m = mi1[0,idx]

        # Calculate new modularity contributions for each row.

        col = B.getrow(idx).T

        if new_m == 1:
            Q_1 = Q.T + col
            Q_0 = Q.T - col
        else:
            Q_1 = Q.T - col
            Q_0 = Q.T + col

        q_idx = w[0,idx] - Q[0,idx]

        Q = np.matrix([Q_1[i,0] if mi1[0, i] == 1 else Q_0[i,0]
                       for i in range(n)])

        Q[0,idx] = q_idx
        Qi = Q.sum() / (2. * mu)

        # Test for new max
        if Qi > maxq:
            maxq = Qi
            # Store new max partition.
            maxqm = mi1.tolist()[0]
            consec_fail = 0

        # Calculate fresh Q vector every 1000 iterations to improve stability.
        if (k+1)%1000 == 0:
            Q_1 = B.dot(mi1.T)
            Q_0 = B.dot(mi0.T)
            Q = np.matrix([Q_1[i,0] if mi1[0,i] == 1 else Q_0[i,0]
                           for i in range(n)])

        # Recalculate change in modularity for each row.
        DQ = w - 2*Q
        DQ = [q if i not in moved else None
              for i, q in enumerate(DQ.tolist()[0])]

        # Check for early termination of algorithm.
        if max_fail is not None and consec_fail > max_fail:

            if maxq > start_q + 10**-15:
                if verbose and len(membership) > 1000:
                    print 'KL loop:', maxq, start_q, len(membership),
                    print sum(maxqm), sum(membership)
     
                return maxqm
     
            else:
                if verbose and len(membership) > 1000:
                    print 'Exiting KL loop', maxq, start_q, len(membership),
                    print sum(maxqm), sum(membership)
     
                return False


    if maxq > start_q + 10**-15:
        if verbose and len(membership) > 1000:
            print 'KL loop:', maxq, start_q, len(membership), sum(maxqm),
            print len(membership)-sum(maxqm)

        return maxqm

    else:
        if verbose and len(membership) > 1000:
            print 'Exiting KL loop', maxq, start_q, len(membership),
            print sum(maxqm), len(membership)-sum(maxqm)
 
        return False


def _ensure_connectivity(g):
    """Reassign vertices not connected to any other community members.

    Ensure the connectivity of two communities using a single igraph Graph
    instance to represent the combined multislice network. Identifies
    vertices not connected to other members of their community and switches
    their membership.

    Args:
        g (igraph.Graph): Represents the combined multislice
            network. Each vertex enters the multislice network one for each
            individual network in which it appears. All edges are undirected
            and indicate either an observed tie or a specified connection
            between the same vertex in different network slices. Must include
            a 'subgraph_memb' vertex attribute and 'subgraph_membership'
            graph attribute to specify the community structure to examine.

    Returns:
        list. Contains each vertex's revised community id.
    """

    name2idx = g['name2idx']
    counter = 0

    connected = False
    tie = False

    while not connected:
        connected = True
        membership = g.vs['subgraph_memb']
        cids = list(set(membership))
        cids.sort()

        for c in [0, 1]:
            vids = [v.index for v in g.vs if v['subgraph_memb'] == c]
            subg = g.subgraph(vids)
            comps = subg.clusters()

            # Identify an unconnected community
            if len(comps) > 1:
                main_size = max([len(sg.vs) for sg in comps.subgraphs()])
                connected = False

                t = 0
                for sg in comps.subgraphs():
                    if len(sg.vs) == main_size:
                        t+=1

                if t > 1:
                    tie = True

                # Switch community for all vertices not in main component
                for sg in comps.subgraphs():
                    if len(sg.vs)!=main_size:
                        for v in sg.vs:
                            m = membership[name2idx[v['name']]]
                            g.vs[name2idx[v['name']]]['subgraph_memb'] = 1-m
                            membership[name2idx[v['name']]] = 1-m
                    elif tie:
                        for v in sg.vs:
                            m = membership[name2idx[v['name']]]
                            g.vs[name2idx[v['name']]]['subgraph_memb'] = 1-m
                            membership[name2idx[v['name']]] = 1-m
                        tie = False

            vids = [v.index for v in g.vs if v['subgraph_memb'] == c]
            subg = g.subgraph(vids)
            comps = subg.clusters()

        # break loop after 100 iterations, indicating a disconnected input
        counter += 1
        if counter > 100:
            connected = True
            print 'Warning: loop limit reached. To avoid disconnected'
            print 'communities, use the '

    return membership


def make_multislice_graph(net_list, w):
    """Makes a multislice representation of a list of separate networks.

    Creates a single network object representing the specified multislice
    structure. Every vertex appears once for each network where it is present.
    Multislice connections occur between different instances of each vertex
    across networks as specified by w.

    Args:
        net_list (list): Contains igraph.Graph instances. Each graph may have a
            'resolution' attribute, which defaults to 1.0 if not specified. The
            multislice network jointly defined by the graphs and the w
            parameters should have exactly one component.

        w: Float or dictionary of the form::

            {(i, j): interslice_weight for i, j in
            itertools.permutation(range(len(net_list)), 2)}

    Returns:
        igraph.Graph. Represents the combined multislice network.
        Each vertex enters the multislice network once for each network in which
        it appears. All edges are undirected and indicate either an observed
        tie or a specified connection between the same vertex in different
        network slices.
    """

    vlist = []
    elist = []
    names = set()

    n = 0
    for i, g in enumerate(net_list):
        for v in g.vs:
            vlist.append(v['name'] + '_' + str(i))
            names.add(v['name'])
        for e in g.es:
            elist.append((e.source + n, e.target + n))
        n += len(g.vs)

    g = ig.Graph()
    g.add_vertices(vlist)

    name2idx = {v['name']: v.index for v in g.vs}
    names = list(names)

    if isinstance(w, float) and w!=0.:

        for n in names:
            interslice = {}

            for i in range(len(net_list)):
                if n + '_' + str(i) in name2idx:
                    interslice[i] = name2idx[n + '_' + str(i)]

            if interslice:
                interslc_edges = itertools.combinations(interslice.items(), 2)
                for (s_net, s_idx), (t_net, t_idx) in interslc_edges:
                    elist.append((s_idx, t_idx))

    elif isinstance(w, dict):

        for n in names:
            interslice = {}

            for i in range(len(net_list)):
                if n + '_' + str(i) in name2idx:
                    interslice[i] = name2idx[n + '_' + str(i)]

            if interslice:
                interslc_edges = itertools.combinations(interslice.items(), 2)
                for (s_net, s_idx), (t_net, t_idx) in interslc_edges:
                    try:
                        if w[(s_net, t_net)] or w[(t_net, s_net)]:
                            elist.append((s_idx, t_idx))
                    except KeyError:
                        pass

    g.add_edges(elist)
    g['name2idx'] = name2idx

    return g
