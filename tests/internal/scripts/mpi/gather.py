# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_gather
from pyccel.stdlib.internal.mpi import MPI_INTEGER8

import numpy as np

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = np.int32(-1)
size = np.int32(-1)
rank = np.int32(-1)

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

master    = np.int32(1)
nb_values = 8

block_length = nb_values // size

# ...
values = np.zeros(block_length, 'int')
for i in range(0, block_length):
    values[i] = 1000 + rank*nb_values + i

print('I, process ', rank, 'sent my values array : ', values)
# ...

# ...
data = np.zeros(nb_values, 'int')

mpi_gather (values, block_length, MPI_INTEGER8,
            data,   block_length, MPI_INTEGER8,
            master, comm, ierr)
# ...

if rank == master:
    print('I, process ', rank, ', received ', data, ' of process ', master)

mpi_finalize(ierr)
