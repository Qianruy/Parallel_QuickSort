#!/usr/bin/env python3
import os
from random import *

in_fname, ou_fname = 'input_test.txt', 'output_test.txt'
np = 6

n = randint(10**4, 10**5)
n = 10**6
# arr = [randint(-10**9, 10**9) for _ in range(n)]
arr = [randint(-10**9, 10**9)] * n
with open(in_fname, 'w') as f:
    f.write(f'{n}\n')
    f.write(' '.join(map(str, arr)) + '\n')

os.system(f'make && mpirun -np {np} ./pqsort {in_fname} {ou_fname} > /dev/null')

with open(ou_fname, 'r') as f:
    prog_out = list(map(int, f.readline().split()))

assert(prog_out == sorted(arr))

os.remove(in_fname)
os.remove(ou_fname)
