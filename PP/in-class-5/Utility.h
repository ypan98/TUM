#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>

#define TEST_PIX_NUM 100
#define MAT_SIZE 1024
#define ITER_NUM 1000

/*
 * This function outputs the execution time. 
 */
class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    float toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

/*
 * Initializes seed for randomized testing.
 */
static unsigned int readInput()
{
    std::cout << "READY" << std::endl;
    unsigned int seed = 0;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if (seed == 0)
    {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    return seed;
}

/*
 * Generates random tests.
 */
static void
generate_test(unsigned int seed, float *problem, float *vec)
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.

    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++)
    {
        problem[i] = ((float)generator() / generator.max());
    }

    float sum = 0;
    for (int i = 0; i < MAT_SIZE; i++)
    {
        vec[i] = ((float)generator() / generator.max());
        sum += vec[i];
    }
}

/*
 * This function outputs the result. 
 */
static void outputResult(unsigned int seed, float *result)
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.
    float sum =0;
    for (int i = 0; i < TEST_PIX_NUM; i++)
    {
        int test_pix_loc = generator() % (MAT_SIZE);
        sum += result[test_pix_loc];
    }
    std::cout << sum << std::endl;

    std::cout << "DONE" << std::endl;
}

#endif // UTILITY_H
