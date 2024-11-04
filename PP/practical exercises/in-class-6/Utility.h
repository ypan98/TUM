#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <random>
#include <chrono>

#define TEST_PIX_NUM 100
#define PROC_NUM 8
#define MAP_WIDTH 1024
#define MAP_HEIGHT (MAP_WIDTH * PROC_NUM)
#define ITER_NUM 100

#define MAX_WAVE_HEIGHT 4.0f

/**
 *
 */
void swap_ptr(float*& ptr1, float*& ptr2){
    float *tmp = ptr1;
    ptr1 = ptr2;
    ptr2 = tmp;
}

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
generate_test(unsigned int seed, float *wave_map)
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.

    for (size_t i = 0; i < MAP_HEIGHT * MAP_WIDTH; i++)
    {
        wave_map[i] = static_cast<float>(generator()) * MAX_WAVE_HEIGHT / static_cast<float>(generator.max());
    }
}

/*
 * This function outputs the result. 
 */
static void outputResult(float *highest_result)
{
    // print highest wave height at each iteration
    std::cout << highest_result[ITER_NUM-1] << "\n";

    std::cout << "DONE" << std::endl;
}

#endif // UTILITY_H
