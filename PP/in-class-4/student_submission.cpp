#include "Utility.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "omp.h"

// HINT: We limit the execution to a single socket for this assignment
// so we don't run into NUMA issues. If you're curious, try to play
// around with these settings and see what happens!
// More info:
// - OMP_PLACES: https://www.openmp.org/spec-html/5.0/openmpse53.html
// - OMP_PROC_BIND: https://www.openmp.org/spec-html/5.0/openmpse52.html
// - proc_bind values: https://www.openmp.org/spec-html/5.0/openmpsu36.html#x56-900002.6.2
// - OMP_NUM_THREADS: https://www.openmp.org/spec-html/5.0/openmpse50.html

// !submission_env OMP_PLACES=sockets
// !submission_env OMP_PROC_BIND=master
// !submission_env OMP_NUM_THREADS=16

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1

// HINT: this helps you with the assignment.
// More info: https://en.cppreference.com/w/cpp/language/storage_duration
thread_local Integer partial_result{0};

/* the mechanism of this function
 *       b
 *     X a
 *    -----
 *
 *
 */
Integer mulInteger(const Integer &a, const Integer &b)
{

    Integer result{0};

    for (size_t i = 0; i < a.size(); i++)
    {
        partial_result = addInteger(mulShiftedInteger(b, a[i], i), partial_result);
    }

    result = addInteger(partial_result, result);

    return result;
}

int main()
{
    unsigned int seed = readInput();
    int problem[NUM_FACTORS];

#ifdef PRINT_TIME
    TicToc total_time;
#endif

    generate_test(seed, problem);
    Integer a = calcProduct(problem);

    outputResult(a);
#ifdef PRINT_TIME
    std::cerr << "time used: " << total_time.toc() << "ms.\n";
#endif
}
