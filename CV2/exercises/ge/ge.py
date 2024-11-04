import numpy as np
import sys

def format_float(f):
    if type(f) is int or f.is_integer():
        return int(f)
    else:
        return f
    
def float_to_str(f) -> str:
    if type(f) is int or f.is_integer():
        return str(int(f))
    else:
        decimal_places = len(str(f).split('.')[1])
        if decimal_places > 15:
            decimal_places = 20
        return "{:.{}f}".format(f, decimal_places)

def gaussian_elimination(matrix):
    operations = []
    n = matrix.shape[0]
    aug = np.hstack((matrix, np.eye(n)))

    for i in range(n):
        # check if zero => swap
        if not aug[i][i]:
            for j in range(i+1, n):
                if aug[j][i]:
                    aug[[i, j]] = aug[[j, i]]
                    operations.append(f"S {i} {j}")
                    break
            # else:
            #     return operations, None
            
        # if not 0
        if aug[i][i]:
            # if not 1 => div by itself
            if aug[i][i] != 1.0:
                coef = 1/aug[i][i]
                coef = format_float(coef)
                aug[i] *= coef
                operations.append(f"M {i} {float_to_str(coef)}")
            # clear column above and below 
            for j in range(n):
                if j != i:
                    if aug[j][i]:
                        coef = -aug[j][i]
                        coef = format_float(coef)
                        aug[j] = aug[j] + coef * aug[i]
                        operations.append(f"A {j} {i} {float_to_str(coef)}")

    # check if degenerate
    for i in range(n):
        if np.allclose(aug[i][i], 0.0):
            return operations, None
    return operations, aug[:, n:]

# read input
n = int(input())
a = np.zeros((n, n))
for i in range(n):
    a[i] = input().split()

operations, null_space = gaussian_elimination(a)
for operation in operations:
    print(operation)
if null_space is not None:
    print('SOLUTION')
    for row in null_space:
        row = list(map(float_to_str, row))
        print(' ' + ' '.join(map(str, row)))
else:
    print('DEGENERATE')