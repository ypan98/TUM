# In-class exercise 4: Help to parallelize big integer multiplication
We are preparing a big time breach closing operation, together with our colleagues all around the solar system. And as always, we need a lot of stored time. So we are asking for your help with optimizing big integer multiplication.    
In the provided snippet you will see the new Integer datatype, which is a vector of ints that represent the digits of our big integer.
For this new datatype, the mulInteger function was created which multiplies 2 Integers. It is your job to parallelize this function
using OpenMP tasks. Also, as usual, do not forget about data racing!


## Your tasks
1. Add OpenMP pragmas to **student_submission.cpp** to achieve parallelism.
2. Submit your code written in **student_submission.cpp** to [https://parprog.caps.in.tum.de/](https://parprog.caps.in.tum.de/).
3. Make sure you reach a speedup of 3.

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
