"""
Please first finish step2 and then copy the `distribution_methods` directory under step2 into this directory.
Then, ideally, you do NOT need to change any other code in this file. 

After you implement all functions in "./taxonomy_organization/graph2dag.py", you generate evaluate results by running:
    python eval_e2e.py

"""
import networkx as nx
from itertools import combinations
from taxonomy_organization.graph2dag import *
from distribution_methods.model import DIHModel


def read_vocab(file_path):
    entityID2entityPreferredName = {}
    entityID2AllNames = {}
    with open(file_path, "r") as fin:    
        for lid, line in enumerate(fin):
            if lid == 0:  # skip header
                continue
            line = line.strip()
            if line:
                segs = line.split("\t")
                entityID = segs[0]
                entityPreferredName = segs[1]
                entityAllNames = [" ".join(cname.split("_")) for cname in segs[2].split("||")]
                entityID2entityPreferredName[entityID] = entityPreferredName
                entityID2AllNames[entityID] = entityAllNames
    return entityID2entityPreferredName, entityID2AllNames


def edge_metrics(T_true, T_pred, verbose=False):
    edges_gold = set(T_true.edges())
    edges_pred = set(T_pred.edges())
    edges_common = edges_gold & edges_pred
    precision = len(edges_common) / len(edges_pred)
    recall = len(edges_common) / len(edges_gold)
    f1 = 2.0*precision*recall/(precision+recall)
    if verbose:
        print(f"edge precision: {precision} ({len(edges_common)}/{len(edges_pred)})")
        print(f"edge recall: {recall} ({len(edges_common)}/{len(edges_gold)})")
        print(f"edge f1: {f1}")
    
    return {"edge_p": precision, "edge_r": recall, "edge_f1": f1}


def ancestor_metrics(T_true, T_pred, verbose=False):
    edges_gold = set(nx.dag.transitive_closure(T_true).edges())
    edges_pred = set(nx.dag.transitive_closure(T_pred).edges())
    edges_common = edges_gold & edges_pred
    precision = len(edges_common) / len(edges_pred)
    recall = len(edges_common) / len(edges_gold)
    f1 = 2.0*precision*recall/(precision+recall)
    if verbose:
        print(f"ancestor precision: {precision} ({len(edges_common)}/{len(edges_pred)})")
        print(f"ancestor recall: {recall} ({len(edges_common)}/{len(edges_gold)})")
        print(f"ancestor f1: {f1}")
    
    return {"ancestor_p": precision, "ancestor_r": recall, "ancestor_f1": f1}


def construct_taxonomy(termID_list, dih, dih_metric_name="invCL", taxorg_method="NoCyc", 
                       graph_init_threshold=0.01, verbose=True):
    """
    main function for construct taxonomy
    
    Args:
        termID_list: list(str). A list of entity ids
        dih: DIHModel object
        dih_metric_name: str. Name of used DIH metric
        taxorg_method: str. Name of taxonomy organization method
        verbose: boolean. Whether to verbosely print out running information
    
    """
    assert dih_metric_name in ["weeds_prec", "clarkeDE", "invCL"]
    assert taxorg_method in ["NoCyc", "DMST"]

    # DIH
    weighted_edges = []  # (hyper, hypo, score)
    for termID_pair in tqdm(list(combinations(termID_list, r=2))):
        score = 2.0 * dih.predict(termID_pair[0], termID_pair[1], dih_metric_name)
        if score >= graph_init_threshold:
            weighted_edges.append([termID_pair[1], termID_pair[0], score])

        # reverse order testing
        score = 2.0 * dih.predict(termID_pair[1], termID_pair[0], dih_metric_name)
        if score >= graph_init_threshold:
            weighted_edges.append([termID_pair[0], termID_pair[1], score])

    # Taxonomy organization
    G = construct_graph(termID_list, weighted_edges)
    if verbose:
        print("Before Taxonomy Organization")
        describe_graph(G)
        print("="*89)

    if taxorg_method == "NoCyc":
        T = NoCyc(G)
    elif taxorg_method == "DMST":
        T = DMST(G)

    if verbose:
        print("After Taxonomy Organization")
        describe_graph(T)
    
    return T

# read in vocabulary 
entityID2entityPreferredName, entityID2AllNames = read_vocab("../raw_data/vocab.txt")
entityName2ID = {}
for k,v in entityID2AllNames.items():
    for t in v:
        t = "_".join(t.split(" "))
        entityName2ID[t] = k

# read in ground truth taxonomy
edges = []
with open("./cs_taxonomy.txt", "r") as fin:
    for line in fin:
        line = line.strip()
        if line:
            segs = line.split("\t")
            edges.append([entityName2ID[segs[0]], entityName2ID[segs[1]]])

T_true = nx.DiGraph()
T_true.add_edges_from(edges)
termID_list = list(T_true.nodes())
    
# construct and evaluate taxonomy
file_out = "./final_e2e_results.tsv"
DIH_METRIC_NAMES = ["weeds_prec", "clarkeDE", "invCL"]
TAXORG_METHODS = ["NoCyc", "DMST"]
GRAPH_THRESHOLDS = [0.01]
CONTEXT_SPACES = ["../step2/entityID2contexts.txt"]

with open(file_out, "w") as fout:
    fout.write("\t".join(["Context_space", "DIH_metric", "Taxorg_method", "Graph_threshold", 
                            "edge_p", "edge_r", "edge_f1", "ancestor_p", "ancestor_r", "ancetor_f1", "\n"]))
    for context_space in CONTEXT_SPACES:
        dih_model = DIHModel(context_space)
        for dih_metric_name in DIH_METRIC_NAMES:
            for taxorg_method in TAXORG_METHODS:
                for graph_threshold in GRAPH_THRESHOLDS:
                    print(f"context_space: {context_space}")
                    print(f"dih_metric_name: {dih_metric_name}, taxorg_method: {taxorg_method}, graph_threshold: {graph_threshold}")
                    try:
                        T_pred = construct_taxonomy(termID_list, dih_model, dih_metric_name, taxorg_method, graph_threshold, False)
                        edge_result = edge_metrics(T_true, T_pred, True)
                        ancestor_result = ancestor_metrics(T_true, T_pred, True)
                        fout.write("\t".join([context_space, dih_metric_name, taxorg_method, str(graph_threshold), 
                                            str(edge_result["edge_p"]), str(edge_result["edge_r"]), str(edge_result["edge_f1"]), 
                                            str(ancestor_result["ancestor_p"]), str(ancestor_result["ancestor_r"]), str(ancestor_result["ancestor_f1"]), "\n"]))
                    except:
                        print("!!! ERROR !!! ATTENTION !!!")
                        fout.write("\t".join([context_space, dih_metric_name, taxorg_method, graph_threshold, 
                                            "-1", "-1", "-1", "-1", "-1", "-1", "\n"]))
                    print("="*89)
