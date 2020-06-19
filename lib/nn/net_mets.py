from lib.defaults import *
from lib.nn.nn_sym_lib import reduced_label
import lib.nn.nnstate as nnstate
from lib.nn.nnstate import update_met_log

def METS_TO_USE(): return [accuracy, matthews_correlation_coefficient, mcc_multi, fill_cmat]

def mcc_basic(TP, FP, TN, FN):
    denom = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denom == 0: denom = 1
    rrr = (TP * TN - FP * FN) / denom
    return rrr

def mcc_multi(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, mcc_multi)
    if TP is not None:
        err('this should never happen if i have >2 classes')
        rrr = mcc_basic(TP, FP, TN, FN)
    elif rrr == _NON_BINARY:
        y_true, y_pred = prep_ys(y_true, y_pred)
        import sklearn.metrics
        # https://github.com/scikit-learn/scikit-learn/issues/16924
        go = True
        for i in range(nnstate.num_pred_classes()):
            if count_nonzero(y_pred == i) == 0:
                go = False
                break
        if go:
            rrr = sklearn.metrics.matthews_corrcoef(y_true, y_pred)

        else:
            rrr = -6

    return update_met_log(mcc_multi, rrr, inc=True)

# (MCC)
def matthews_correlation_coefficient(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, matthews_correlation_coefficient)
    if TP is not None:
        rrr = mcc_basic(TP, FP, TN, FN)
    elif rrr == _NON_BINARY:
        y_true, y_pred = prep_ys(y_true, y_pred)

        true_class_names = inv_map(nnstate.CURRENT_TRUE_MAP)

        reduced_y_true = []
        for y in y_true:
            tru_nam = true_class_names[int(y)]
            reduced_y_true.append(reduced_label(tru_nam))

        reduced_y_pred = []
        for y in y_pred:
            tru_nam = true_class_names[int(y)]
            reduced_y_pred.append(reduced_label(tru_nam))

        TP, FP, TN, FN, P, N = binary_results(reduced_y_true, reduced_y_pred)
        if TP is not None:
            rrr = mcc_basic(TP, FP, TN, FN)

    return update_met_log(matthews_correlation_coefficient, rrr, inc=True)



def accuracy(y_true, y_pred):
    rrr = 1 - error_rate(y_true, y_pred, real_error=False)
    return update_met_log(accuracy, rrr)

def error_rate_basic(FP, FN, P, N):
    return (FP + FN) / (P + N)


def error_rate(y_true, y_pred, real_error=True):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, error_rate)
    if TP is not None:
        if (P + N) == 0:
            rrr = _EMPTY_TENSOR
        else:
            rrr = error_rate_basic(FP, FN, P, N)
    elif rrr == _NON_BINARY:
        y_true, y_pred = prep_ys(y_true, y_pred)
        rrr = count_nonzero(y_pred != y_true) / len(y_pred)
    return update_met_log(error_rate, rrr) if real_error else rrr

# recall (REC), true positive rate (TPR)
# (SN), best 1.0, worst 0.0
def sensitivity(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, sensitivity)
    if TP is not None:
        if P == 0:
            rrr = _EMPTY_TENSOR
        else:
            rrr = TP / P
    return update_met_log(sensitivity, rrr)

# (PREC), best 1.0, worst 0.0
def precision(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, precision)
    if TP is not None:
        if (TP + FP) == 0:
            rrr = _EMPTY_TENSOR
        else:
            rrr = TP / (TP + FP)
    return update_met_log(precision, rrr)

#  (FPR)
def false_positive_rate(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, false_positive_rate)
    if TP is not None:
        if N == 0:
            rrr = _EMPTY_TENSOR
        else:
            rrr = FP / N
    return update_met_log(false_positive_rate, rrr)

# (SP).  (true negative rate_
def specificity(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, specificity)
    if TP is not None:
        if N == 0:
            rrr = - 2
        else:
            rrr = TN / N
    return update_met_log(specificity, rrr)

# (MR), best 0.0, worst 1.0, converse of hit rate
def miss_rate(y_true, y_pred):
    rrr, TP, FP, TN, FN, P, N = basics(y_true, y_pred, miss_rate)
    if TP is not None:
        if P == 0:
            rrr = _EMPTY_TENSOR
        else:
            rrr = FN / P
    return update_met_log(miss_rate, rrr)

cmat = arr2d()
batch_count = 0
total_steps = None
batch_sub_count = None
def fill_cmat(y_true, y_pred):
    [inc(cmat, (pred, tru)) for tru, pred in zip(*prep_ys(y_true, y_pred))]
    global batch_count, total_steps, batch_sub_count
    if batch_sub_count is not None:
        batch_sub_count += 1
    if batch_sub_count is None or batch_sub_count == 3:
        log(f'Finished {batch_count}/{total_steps} steps')
        batch_count += 1
        if batch_sub_count == 3:
            batch_sub_count = 1
    return 0


def prep_ys(y_true, y_pred):
    if not isinstance(y_true, mparray):
        y_true = y_true.numpy()
        y_true = y_true[:, 0]
    if not isinstance(y_pred, mparray):
        y_pred = y_pred.numpy()
    y_pred = y_pred[:, 0:nnstate.num_pred_classes()]
    y_pred = arr(list(map(maxindex, y_pred)))
    return y_true, y_pred



def binary_results(y_true, y_pred):
    y_true = arr(y_true)
    y_pred = arr(y_pred)
    if any(arr(y_true) > 1) or any(arr(y_pred) > 1):
        err('binary results cannot be done when there are more than two classes')
    neg = 0
    pos = 1
    P = count_nonzero(y_true == pos)
    N = count_nonzero(y_true == neg)
    TP = count_nonzero(bitwise_and(y_pred == pos, y_true == pos))
    FP = count_nonzero(bitwise_and(y_pred == pos, y_true == neg))
    TN = count_nonzero(bitwise_and(y_pred == neg, y_true == neg))
    FN = count_nonzero(bitwise_and(y_pred == neg, y_true == pos))
    return TP, FP, TN, FN, P, N

def basics(y_true, y_pred, fun):
    rrr = _DEFAULT_RESULT
    P = None
    N = None
    TP = None
    FP = None
    TN = None
    FN = None
    if empty_tensor(y_true):
        rrr = _EMPTY_TENSOR
    elif unused_metric(fun):
        rrr = _DO_NOT_USE
    elif nnstate.num_pred_classes() > 2:
        rrr = _NON_BINARY
    else:
        y_true, y_pred = prep_ys(y_true, y_pred)
        TP, FP, TN, FN, P, N = binary_results(y_true, y_pred)
    return rrr, TP, FP, TN, FN, P, N

def unused_metric(fun):
    return fun.__name__ not in [x.__name__ for x in
                                METS_TO_USE()]

def empty_tensor(y_true): return y_true.shape[0] is None or (
        y_true.shape[0].__class__.__name__ == 'Dimension' and y_true.shape[0].value is None)

_DEFAULT_RESULT = -3
_EMPTY_TENSOR = -2
_DO_NOT_USE = -4
_NON_BINARY = -5
