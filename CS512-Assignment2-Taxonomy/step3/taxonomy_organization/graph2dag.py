"""
You only need to implement two functions in this file: 
    (1) NoCyc: which iteratively removes one (carefully selected) edge from each cycle
    (2) DMST: directed maximum-spanning-tree using optimal branching algorithm
"""
import networkx as nx


def read_edges_file(file_name):
    edge_w_weights = []
    with open(file_name, "r") as fin:
        for line_id, line in enumerate(fin):
            if line_id == 0:  # skip_header
                continue
            else:
                segs = line.split("\t")
                edge_w_weights.append([segs[0], segs[1]]+[float(s) for s in segs[2:]])
    return edge_w_weights


def construct_graph(nodes, weighted_edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    weighted_edges = [(ele[0], ele[1], {'weight': ele[2]}) for ele in weighted_edges]
    G.add_edges_from(weighted_edges)
    return G


def describe_graph(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"is DAG?: {nx.is_directed_acyclic_graph(G)}")
    print(f"is Tree?: {nx.is_tree(G)}")
    print(f"is Forest?: {nx.is_forest(G)}")
    if G.is_directed():
        print("G is a directed graph")
        print(f"\tNumber of weakly connected components: {nx.number_weakly_connected_components(G)}")
        print(f"\tNumber of strongly connected components: {nx.number_strongly_connected_components(G)}")


def NoCyc(G):
    G_copy = G.copy()

    ### YOUR CODE HERE
    # 
    ### END YOUR CODE
    return G_copy


def DMST(G):
    G_copy = G.copy()

    ### YOUR CODE HERE
    # 
    ### END YOUR CODE
    return G_copy
