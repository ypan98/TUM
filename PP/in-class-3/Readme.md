# In-class exercise 3: Help to parallelize K-NN
There were no new time breaches this week.
But we received a corporate order which will help us pay out salaries and save time for future time breaches.
A company called Bambridge Neolitica has a Facebook test app, which predicts a Hogwarts House for the user based on their answers.
They are making this prediction using the K-Nearest-Neighbors algorithm. You have to help us optimize/parallelize it.
We have a set of data points which is used for the predictions and a set of input points for which we have to make these predictions. All points are assumed to be in 2D. Each point in the data point cloud has a class label (for ex. Gryffindor - 0, Slytherin - 1). The algorithm is executed as follows:

1. For a point P in the input point cloud, we find the k nearest points in the data point cloud.
2. Count how many times each class appears among the k nearest neighbors. Assign the class label which appears the most to point P.
3. Repeat step 1 and step 2 for every point in the input point cloud.
4. Count the number of points in each class for the input point cloud and print.

## Your tasks
1. Add OpenMP pragmas to **student_submission.cpp** to achieve parallelism.
2. Submit your code written in **student_submission.cpp** to [https://parprog.caps.in.tum.de/](https://parprog.caps.in.tum.de/).
3. Make sure you reach a speedup of 15.

Remember - you only have 10 minutes.

## Technical details
The files Utility.h and Makefile were already sent to the server. You can only modify student_submission.cpp (in the beginning identical to sequential_implementation.cpp). As soon as you have optimized your code, upload ONLY student_submission.cpp to the server and check if your message was sent successfully.

Good luck! 

## How to run the code

```bash
make
# run sequential implementation
./sequential_implementation
# run your solution
./student_implementation
```
