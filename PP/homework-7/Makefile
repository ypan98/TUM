CXX = mpicxx
CXX_FLAGS = --std=c++17 -Wall -Wextra -march=native -O3 -g -DOMPI_SKIP_MPICXX
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gcc 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157

all: env_file sequential_implementation student_submission

Utility.o: Utility.cpp Utility.h
	$(CXX) $(CXX_FLAGS) -c -o Utility.o Utility.cpp

VideoOutput.o: VideoOutput.cpp VideoOutput.h
	$(CXX) $(CXX_FLAGS) -c -o VideoOutput.o VideoOutput.cpp

lib/TinyPngOut.o: lib/TinyPngOut.hpp lib/TinyPngOut.cpp
	$(CXX) $(CXX_FLAGS) -c -o lib/TinyPngOut.o lib/TinyPngOut.cpp

sequential_implementation: life.h Utility.o VideoOutput.o sequential_implementation.cpp lib/TinyPngOut.o
	$(CXX) $(CXX_FLAGS) -o sequential_implementation VideoOutput.o Utility.o lib/TinyPngOut.o sequential_implementation.cpp

student_submission: life.h Utility.o VideoOutput.o student_submission.cpp lib/TinyPngOut.o
	$(CXX) $(CXX_FLAGS) -o student_submission VideoOutput.o Utility.o lib/TinyPngOut.o student_submission.cpp

env_file: student_submission.cpp
	sed -n -e 's/.*!submission_env \(.*\)/\1/p' student_submission.cpp > student_submission.env

clean:
	rm -f sequential_implementation student_submission *.o lib/*.o out.mp4 *.png student_submission.env
