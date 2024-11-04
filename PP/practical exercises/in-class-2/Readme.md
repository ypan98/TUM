# In-class exercise 2: Help the starship detect damage
We have received an urgent request from our colleagues from Proxima B. Their Timekeepers team is understaffed and they need our help to save some time. Thus you have to optimize one of their programs. You are provided with a code snippet which is used by starships in the Proxima Centauri system to avoid being hit and detect damage from meteorites. Proxima B has multicore processors installed on their starships, so you are welcome to use threads to achieve maximum speedup.

## Your tasks
1. Fill in the code in **student_submission.cpp** to optimize the damage detection process.
2. Submit your code written in **student_submission.cpp** to [https://parprog.caps.in.tum.de/](https://parprog.caps.in.tum.de/).
3. Make sure you reach a speedup of 12.

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
