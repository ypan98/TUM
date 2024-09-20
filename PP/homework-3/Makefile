CXX = g++
CXX_FLAGS = --std=c++17 -Wall -Wextra -march=native -O3
LIBS = -lcrypto

all: sequential_implementation student_submission env_file

Utility.o: Utility.cpp
	$(CXX) $(CXX_FLAGS) -c -o Utility.o Utility.cpp $(LIBS)

sequential_implementation: sequential_implementation.cpp Utility.o
	$(CXX) $(CXX_FLAGS) -o sequential_implementation sequential_implementation.cpp Utility.o $(LIBS)

student_submission: student_submission.cpp Utility.o
	$(CXX) $(CXX_FLAGS) -o student_submission student_submission.cpp Utility.o $(LIBS) -pthread

env_file: student_submission.cpp
	sed -n -e 's/.*!submission_env \(.*\)/\1/p' student_submission.cpp > student_submission.env

clean:
	rm -f sequential_implementation student_submission *.o student_submission.env
