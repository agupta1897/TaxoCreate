"""
You do NOT need to change any code in this file. 
"""

import numpy as np
import logging
import gzip
import scipy.sparse as sp

try:
    import cPickle as pickle
except ImportError:
    import pickle


def __try_three_columns(string):
    fields = string.split("\t")
    if len(fields) > 3:
        fields = fields[:3]
    if len(fields) == 3:
        return fields[0], fields[1], float(fields[2])
    if len(fields) == 2:
        return fields[0], fields[1], 1.0
    else:
        raise ValueError("Invalid number of fields {}".format(len(fields)))


def __load_sparse_matrix(filename, same_vocab):
    """
    Actual workhorse for loading a sparse matrix. See docstring for
    read_sparse_matrix.

    """
    objects = ["<OOV>"]
    rowvocab = {"<OOV>": 0}
    if same_vocab:
        colvocab = rowvocab
    else:
        colvocab = {}
    _is = []
    _js = []
    _vs = []

    # Read gzip files
    if filename.endswith(".gz"):
        f = gzip.open(filename, "r")
    else:
        f = open(filename, "rb")

    for line in f:
        line = line.decode("utf-8")
        target, context, weight = __try_three_columns(line)
        if target not in rowvocab:
            rowvocab[target] = len(rowvocab)
            objects.append(target)
        if context not in colvocab:
            colvocab[context] = len(colvocab)
            if same_vocab:
                objects.append(context)

        _is.append(rowvocab[target])
        _js.append(colvocab[context])
        _vs.append(weight)

    # clean up
    f.close()

    _shape = (len(rowvocab), len(colvocab))
    spmatrix = sp.csr_matrix((_vs, (_is, _js)), shape=_shape, dtype=np.float64)
    return spmatrix, objects, rowvocab, colvocab


def read_sparse_matrix(filename, same_vocab=False):
    """
    Reads in a 3 column file as a sparse matrix, where each line (x, y, v)
    gives the name of the row x, column y, and the value v.

    If filename ends with pkl, will assume the file is a pickle cache file.
    Else, will assume this file is 3 column txt/tsv files and generate the cache
        on the fly.
    
    Args:
        filename: str. The filename containing sparse matrix in 3-col format.
        same_vocab: bool. Indicates whether rows and columns have the same vocab.

    Returns:
        A tuple containing (spmatrix, id2row, row2id, col2id):
            spmatrix: a scipy.sparse matrix with the entries
            id2row: a list[str] containing the names for the rows of the matrix
            row2id: a dict[str,int] mapping words to row indices
            col2id: a dict[str,int] mapping words to col indices. If same_vocab,
                this is identical to row2id.
    """
    if filename.endswith(".pkl"):  # read from cache
        logging.info("Using space cache {}".format(filename))
        with open(filename, "rb") as pklf:
            return pickle.load(pklf)
    else:  # read from tsv file and generate cache
        logging.info("Loading text feature file {}".format(filename))
        result = __load_sparse_matrix(filename, same_vocab=same_vocab)
        logging.info("Dumping the binary cache {}.pkl".format(filename))
        with open(filename + ".pkl", "wb") as pklf:
            pickle.dump(result, pklf)
        return result
