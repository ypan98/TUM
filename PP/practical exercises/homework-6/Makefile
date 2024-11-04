CXX = g++
CXX_FLAGS = --std=c++17 -Wall -Wextra -mavx -O3 -fstrict-overflow -fno-tree-vectorize

all: env_file sequential_implementation student_submission

sequential_implementation: dgemm.h sequential_implementation.cpp
	$(CXX) $(CXX_FLAGS) -o sequential_implementation sequential_implementation.cpp

student_submission: dgemm.h student_submission.cpp
	$(CXX) $(CXX_FLAGS) -o student_submission student_submission.cpp

env_file: student_submission.cpp
	sed -n -e 's/.*!submission_env \(.*\)/\1/p' student_submission.cpp > student_submission.env

clean:
	rm -f sequential_implementation student_submission student_submission.env
