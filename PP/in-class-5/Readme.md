# In-class exercise 5: Help parallelize matrix-vector multiplication
We continue to prepare for a big time breach closing operation together with our colleagues all around the solar system. So we still need stored time. We hope you will help us with that by optimizing matrix-vector multiplication.    
In the provided snippet you will see the TODOs you have to implement in order to parallelize the code using Intel Intrinsics vector operations. For this exercise we are using 256 bit vectors and single precision floating-point numbers.


## Your tasks
1. Add Intel Intrinsics vector operations to **student_submission.cpp** to achieve parallelism.
2. Submit your code written in **student_submission.cpp** to [https://parprog.caps.in.tum.de/](https://parprog.caps.in.tum.de/).
3. Make sure you reach a speedup of 4.

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
