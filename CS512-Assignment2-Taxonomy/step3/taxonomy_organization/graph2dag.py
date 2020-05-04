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
    while True:
        try:
            nx.find_cycle(G_copy)
            for c in nx.strongly_connected_components(G_copy):
                nodeList = []
                for node in G_copy.nodes():
                    if str(node) in c:
                        nodeList.append(node)
            
                subG = G_copy.subgraph(nodeList)
                smallestEdge = ""
                for edge in subG.edges( data = "weight"):
                    if (smallestEdge == "" ):
                        smallestEdge = edge
                    else:
                        if(edge[2] < smallestEdge[2]):
                            smallestEdge = edge
                if (smallestEdge != ""): 
                    G_copy.remove_edge(smallestEdge[0], smallestEdge[1])
        except:
            break
    return G_copy



def DMST(G):
    G_copy = G.copy()
    G_copy.add_node("Dummy")
    for node in G_copy.nodes():
        G_copy.add_edge("Dummy", node, weight = 0.1)


    G_nodir = G_copy.to_undirected()
    T = nx.maximum_spanning_tree(G_nodir)
    T_dir = T.to_directed()
    T_dir.remove_node("Dummy")

    edgeListToRemove = []


    for edge in T_dir.edges:
        removeEdge = True
        for edgeG in G.edges:
            if(str(edge[0]) == str(edgeG[0]) and str(edge[1]) == str(edgeG[1])):
                removeEdge = False
                break
        if (removeEdge == True):
            edgeListToRemove.append( (edge[0], edge[1]) )
        
    for edge in edgeListToRemove:
        T_dir.remove_edge(edge[0], edge[1])
    return T_dir
