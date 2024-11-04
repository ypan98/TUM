import numpy as np

def row_echelon(A):
    A = np.copy(A)
    r, c = A.shape
    if r == 0 or c == 0:
        return A
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        B = row_echelon(A[:,1:])
        return np.hstack([A[:,:1], B])
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row
    A[0] = A[0] / A[0,0]
    A[1:] -= A[0] * A[1:,0:1]
    B = row_echelon(A[1:,1:])
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

def fix(A, free_vars):
    A = np.copy(A)
    for i in range(A.shape[0]):
        for j in free_vars:
            # find one row below 
            if not A[i,j]:
                for k in range(i+1, A.shape[0]):
                    if A[k,j]:
                        A[i] += A[k]
                        break
    return A

def get_pivot_cols(A):
    pivot_cols = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                pivot_cols.append(j)
                break
    return pivot_cols

def null_space(A):
    rref_A = row_echelon(A)
    pivot_cols = get_pivot_cols(rref_A)
    free_vars = [i for i in range(M.shape[1]) if i not in pivot_cols]
    rref_A = fix(rref_A, free_vars)
    basis     = []
    for free_var in free_vars:
        vec           = [0] * A.shape[1]
        vec[free_var] = 1
        for piv_row, piv_col in enumerate(pivot_cols):
            vec[piv_col] -= rref_A[piv_row, free_var]
        basis.append(vec)
    return np.array(basis).T

def give_first_normalized_ns_col(ns, num_a_rows):
    cols = ns.shape[1]
    for j in range(cols):
        col = ns[:,j]
        a = col[:num_a_rows]
        b = col[num_a_rows:]
        if np.sum(a) and np.sum(b):
            a = a / np.sum(a)
            b = b / np.sum(b)
            res = np.concatenate((a,b))
            return res[:,None]
    return None

dim = int(input())
A = []
B = []
for i in range(2):
    n = int(input())
    landmarks = []
    for _ in range(n):
        input_list = input().split()
        coords = list(map(float, input_list[1:]))
        landmarks.append(coords)
    if i == 0:
        A = landmarks
    else:
        B = landmarks

A = np.array(A).reshape(-1, dim)
B = np.array(B).reshape(-1, dim)
M = np.hstack((A.T, -B.T))



ns = null_space(M)
if not len(ns):
    print("N")
else:
    num_a_rows = A.shape[0]
    col = give_first_normalized_ns_col(ns, num_a_rows)
    if col is None:
        print("N")
    else:
        ns_B = col[num_a_rows:]
        mp = np.sum(B*ns_B, axis=0)
        s = "Y"
        for coord in mp:
            if np.isclose(coord, np.round(coord)):
                coord = int(coord)
            s += " " + str(coord)
        print(s)