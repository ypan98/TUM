#include "Utility.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "omp.h"

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1


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
        auto tmp = addInteger(result, mulShiftedInteger(b, a[i], i));
        result = tmp; 
    }
    
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
