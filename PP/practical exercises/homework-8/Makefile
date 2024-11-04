CXX = g++
MPICXX = mpicxx
CXX_FLAGS = -std=c++17 -Wall -Wextra -march=native -g -O3 -fopenmp -Wno-unused -DOMPI_SKIP_MPICXX
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gcc 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157

all: sequential_implementation student_submission

utility: Utility.cpp Utility.h
	$(CXX) $(CXX_FLAGS) -c -o Utility.o Utility.cpp 

sequential_implementation: sequential_implementation.cpp utility
	$(MPICXX) $(CXX_FLAGS) sequential_implementation.cpp Utility.o -o sequential_implementation 

student_submission: student_submission.cpp utility env_file mpirun_options
	$(MPICXX) $(CXX_FLAGS) student_submission.cpp Utility.o -o student_submission

env_file: student_submission.cpp
	sed -n -e 's/.*!submission_env \(.*\)/\1/p' student_submission.cpp > student_submission.env

mpirun_options: student_submission.cpp
	sed -n -E -e 's/.*!mpirun_option ((--bind-to|--map-by|--rank-by)\s\S+).*/\1/p' student_submission.cpp > student_mpirun.opt

clean:
	rm -f sequential_implementation student_submission Utility.o student_submission.env student_mpirun.opt