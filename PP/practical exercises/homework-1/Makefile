CXX = g++
CXX_FLAGS = --std=c++17 -Wall -Wextra -march=native -mavx -O3

all: sequential_implementation student_submission

sequential_implementation: sequential_implementation.cpp
	$(CXX) $(CXX_FLAGS) -o sequential_implementation  sequential_implementation.cpp

student_submission: student_submission.cpp
	$(CXX) $(CXX_FLAGS) -o student_submission student_submission.cpp
