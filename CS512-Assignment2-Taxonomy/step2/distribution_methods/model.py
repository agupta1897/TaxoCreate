"""
You only need to implement three functions in this file: 
    (1) weeds_prec()
    (2) clarkeDE()
    (3) invCL()
"""

import numpy as np
import logging
from tqdm import tqdm
from scipy.stats import entropy
from .reader import read_sparse_matrix


def invCL(x_row, y_row):
    """
    Computes invCL(x, y)

    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.

    Returns:
        float. Estimation of distributional inclusion.
    """
    y_row = y_row.ravel()
    x_row = x_row.ravel()
    a = clarkeDE(x_row, y_row)[0]
    b = clarkeDE(y_row, x_row)[0]
    val = a*(1-b)


    ### YOUR CODE HERE
    return (val**0.5, 0)
    ### END YOUR CODE


def clarkeDE(x_row, y_row):
    """
    clarkeDE similarity

    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.

    Returns:
        float. Estimation of distributional inclusion.
    """

    ### YOUR CODE HERE
    summ = 0
    summ2 = 0
    y_row = y_row.ravel()
    x_row = x_row.ravel()
    for i in range(x_row.shape[0]):
        summ2 = summ2 + x_row[i]
        if(x_row[i] > y_row[i]):
            summ = summ + y_row[i]
        else:
            summ = summ + x_row[i]

    denominator = summ2


    return (summ / (denominator + 1e-12),0)
    ### END CODE HERE


def weeds_prec(x_row, y_row):
    """
    WeedsPrec similarity

    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.

    Returns:
        float. Estimation of distributional inclusion.
    """
    summ = 0
    summ2 = 0
    y_row = y_row.ravel()
    x_row = x_row.ravel()
    for i in range(y_row.shape[0]):
        summ2 = summ2 + x_row[i]
        if(y_row[i] != 0 ):
            summ = summ + x_row[i]

    # HINT: Get the mutual contexts: use y as a binary vector and apply dot product
    # with x: If c is a mutual context, it is 1 in y_non_zero and the value
    # ppmi(x, c) is added to the sum Otherwise, if it is 0 in either x or y, it
    # adds 0 to the sum.

    ### YOUR CODE HERE
    numerator = summ
    denominator = summ2
    A = []
    A.append(numerator / (denominator + 1e-12))
    return A
    ### END CODE HERE

def mdot(x_row, y_row):
    """
    Inner product of x and y

    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.

    Returns:
        float. Estimation of distributional inclusion.
    """
    return (x_row * y_row).sum(axis=1)


def cosine(x_row, y_row):
    """
    Cosine similarity

    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.

    Returns:
        float.
    """
    return mdot(x_row, y_row) / np.sqrt(mdot(x_row, x_row) * mdot(y_row, y_row) + 1e-12)


class DIHModel(object):
    """
    Hypernymy Prediction Model based on Distributional Inclusive Hypothesis (DIH)

    Args:
        space_filename: filename of the 3-column sparse distributional space.
    """
    _row_entropy_cache = {}
    def __init__(self, space_filename):
        logging.info("Prepping sparse matrix")

        self.space_filename = space_filename
        self.matrix, self.objects, self.vocab, self.col_vocab = read_sparse_matrix(
            space_filename
        )
        self.metrics2fns = {
            "weeds_prec": self.__predict_weeds_prec,
            "clarkeDE": self.__predict_clarkeDE,
            "invCL": self.__predict_invCL,
            "cosine": self.__predict_cosine
        }

        # Below parameters are used only for SLQS and SLQS_plus_metric model
        self.is_SLQS_initialized = False
        self.topK = None
        self.colent = None
    
    def init_SLQS(self, topK=1000):
        """
        Args:
            topK: int. The number of entropy items for each row.
        """
        logging.info("First run of SLQS model, computing column entropies")
        self.topK = topK
        tr = self.matrix.transpose().tocsr()

        entropies = []
        # Minibatches for computation efficiency
        bs = 1024
        for idx_start in tqdm(range(0, tr.shape[0], bs), desc="computing row entropies"):
            idx_end = min(idx_start + bs, tr.shape[0])
            v = tr[idx_start:idx_end].todense().A
            entropies += list(entropy(v.T))
        self.colent = np.array(entropies)
        assert len(self.colent) == tr.shape[0]
        logging.info("Done computing entropies")
        # cleanup
        del tr
        self.is_SLQS_initialized = True
        logging.info("Done computing row entropies")

    def __compute_row_entropy(self, i):
        """ Used only for SLQS or SLQS_plus_meaure model
        """
        if i in self._row_entropy_cache:
            return self._row_entropy_cache[i]
        row = self.matrix[i]
        data = row.data
        indx = row.indices
        if len(data) == 0:
            return 0
        k = min(self.topK, len(data))
        ranked = np.argpartition(data, -k)
        sigdims = indx[ranked[-k:]]
        rowent = np.median(self.colent[sigdims])
        self._row_entropy_cache[i] = rowent
        return rowent

    def __predict_weeds_prec(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return weeds_prec(lhs, rhs)[0]

    def __predict_clarkeDE(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return clarkeDE(lhs, rhs)[0]
    
    def __predict_invCL(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return invCL(lhs, rhs)[0]

    def __predict_cosine(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return cosine(lhs, rhs)[0]

    def __predict_SLQS(self, hypo, hyper):
        assert self.is_SLQS_initialized, "Need to initialize SLQS model first by calling init_SLQS()"
        x = self.vocab.get(hypo, 0)
        y = self.vocab.get(hyper, 0)
        ent_x = self.__compute_row_entropy(x)
        ent_y = self.__compute_row_entropy(y)
        return 1 - ent_x / (ent_y + 1e-12)

    def __predict_SLQS_plus_metric(self, hypo, hyper, metric_name_suffix):
        entropy_score = self.__predict_SLQS(hypo, hyper)
        measure_score = self.metrics2fns[metric_name_suffix](hypo, hyper)
        return entropy_score * measure_score

    def predict(self, hypo, hyper, metric_name="weeds_prec"):
        """
        Core modeling procedure, estimating the degree to which hypo is_a hyper.

        Args:
            hypo: str. A hypothesized hyponym.
            hyper: str. A hypothesized hypernym.
            metric_name: str. A DIH metric name

        Returns:
            float. The score estimating the degree to which hypo is_a hyper.
                Higher values indicate a stronger degree.
        """
        if metric_name == "weeds_prec":
            return self.__predict_weeds_prec(hypo, hyper)
        elif metric_name == "clarkeDE":
            return self.__predict_clarkeDE(hypo, hyper)
        elif metric_name == "invCL":
            return self.__predict_invCL(hypo, hyper)
        elif metric_name == "cosine":
            return self.__predict_cosine(hypo, hyper)
        elif metric_name == "SLQS":
            return self.__predict_SLQS(hypo, hyper)
        elif metric_name.startswith("SLQS_plus_"):
            metric_name_suffix = metric_name[10:]
            return self.__predict_SLQS_plus_metric(hypo, hyper, metric_name_suffix)
