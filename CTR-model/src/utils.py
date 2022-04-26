import random
import numpy as np
import torch
import scipy.sparse as sp


def init_seed(seed):
    """ init random seed for random functions in numpy, torch, cuda and cudnn
    :param seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def get_norm_adj_mat(interaction_matrix, n_users, n_items):
    r"""Get the normalized interaction matrix of users and items.
    Construct the square matrix from the training data and normalize it
    using the laplace matrix.
    .. math::
        A_{hat} = D^{-0.5} \times A \times D^{-0.5}
    Returns:
        Sparse tensor of the normalized interaction matrix.
    """
    # build adj matrix
    A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    inter_M = interaction_matrix
    inter_M_t = interaction_matrix.transpose()

    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)

    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    L_2 = L.dot(L)
    L_u = L_2[:n_users, :n_users]
    L_i = L_2[n_users:, n_users:]

    SparseL = convert_sp_to_tensor(L)
    SparseL_u = convert_sp_to_tensor(L_u)
    SparseL_i = convert_sp_to_tensor(L_i)
    return SparseL, SparseL_u , SparseL_i

def convert_sp_to_tensor(sp_mat):
    # covert sparse matrix matrix to tensor
    sp_mat = sp.coo_matrix(sp_mat)
    row = sp_mat.row
    col = sp_mat.col
    i = np.array([row, col])
    i = torch.LongTensor(i)
    data = torch.FloatTensor(sp_mat.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(sp_mat.shape))
    return SparseL

