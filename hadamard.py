import torch
import numpy as np
from scipy.linalg import hadamard
def find_min_power(n):
    return 1 << (n - 1).bit_length()

def grayCode(n):
    # Right Shift the number
    # by 1 taking xor with
    # original number
    #return n ^ (n >> 1)
    return [i^(i>>1) for i in range(n)]

def reverse(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1: result |= 1 << (n - 1 - i)
    return result

def reverseCode(n):
    return [reverse(x, np.log2(n).astype(int)) for x in range(n)]

def permutation(A, axis = -1):
    if axis != -1:
        A = np.transpose(A, -1, axis)
    n = A.shape[-1]
    B = A[..., reverseCode(n)]
    C = B[..., grayCode(n)]
    if axis != -1:
        C = np.transpose(C, -1, axis)
    return C

def permutation_pt(A, axis = -1):
    if axis != -1:
        A = torch.transpose(A, -1, axis)
    n = A.shape[-1]
    B = A[..., reverseCode(n)]
    C = B[..., grayCode(n)]
    if axis != -1:
        C = torch.transpose(C, -1, axis)
    return C

def fwht(u, axis=-1, fast=False):
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        y = permutation_pt(x.squeeze(-2))
    else:
        H = torch.tensor(permutation(hadamard(n)), dtype=torch.float, device=u.device)
        y = u @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y

def ifwht(u, axis=-1, fast=False):
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        y = permutation_pt(x.squeeze(-2)) / n
    else:
        H = torch.tensor(permutation(hadamard(n)), dtype=torch.float, device=u.device)
        y = u @ H / n
    if axis != -1:
        y = torch.transpose(y, -1, axis)

    return y
