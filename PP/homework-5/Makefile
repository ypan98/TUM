CXX = g++
CXX_FLAGS = --std=c++17 -Wall -Wextra -fopenmp -march=native -O3

all: sequential_implementation student_submission env_file

Utility.o: Utility.cpp
	$(CXX) $(CXX_FLAGS) -c -o Utility.o Utility.cpp

VideoOutput.o: VideoOutput.cpp
	$(CXX) $(CXX_FLAGS) -c -o VideoOutput.o VideoOutput.cpp

LodePng.o: lib/lodepng.cpp
	$(CXX) $(CXX_FLAGS) -c -o LodePng.o lib/lodepng.cpp

sequential_implementation: sequential_implementation.cpp Utility.o VideoOutput.o LodePng.o
	$(CXX) $(CXX_FLAGS) -o sequential_implementation sequential_implementation.cpp Utility.o VideoOutput.o LodePng.o

student_submission: student_submission.cpp Utility.o VideoOutput.o LodePng.o
	$(CXX) $(CXX_FLAGS) -o student_submission student_submission.cpp Utility.o VideoOutput.o LodePng.o


env_file: student_submission.cpp
	sed -n -e 's/.*!submission_env \(.*\)/\1/p' student_submission.cpp > student_submission.env

clean:
	rm -f sequential_implementation student_submission *.o out.mp4 student_submission.env
