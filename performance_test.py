#!/usr/bin/env python3
import os
import sys
from random import *

if len(sys.argv) < 2:
    sys.exit(1)

np = int(sys.argv[1])
n = 10**6
arr = [randint(-10**6, 10**6) for _ in range(n)]

in_fname = f'input_{n}.txt'
with open(in_fname, 'w') as f:
        f.write(f'{n}\n')
        f.write(' '.join(map(str, arr)) + '\n')


ou_fname = f'output_{np}.txt'

os.system(f'make && mpirun -np {np} ./pqsort {in_fname} {ou_fname} > /dev/null')

with open(ou_fname, 'r') as f:
    prog_out = list(map(int, f.readline().split()))

# assert(prog_out == sorted(arr))

os.remove(in_fname)
# os.remove(ou_fname)

