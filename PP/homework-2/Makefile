CXX = c++
CXX_FLAGS = --std=c++17 -Wall -Wextra -march=native -O3 -lm

all: sequential_implementation student_submission

student_submission: student_submission.cpp
	$(CXX) $(CXX_FLAGS) -o student_submission -O3 student_submission.cpp -pthread

sequential_implementation: sequential_implementation.cpp
	$(CXX) $(CXX_FLAGS) -o sequential_implementation sequential_implementation.cpp

clean:
	rm -f sequential_implementation student_submission