#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <string.h>
#include <random>
#include <chrono>

#define MAP_SIZE 100
#define ROCKS_NUM 480
#define THREAD_NUM 32
#define NUM_DATAPOINTS 100
#define MAX_X 5.0
#define MAX_VELO 3.0

unsigned int rocks_pos[ROCKS_NUM][2];
double rocks_vel[ROCKS_NUM][2];
double datapoints[NUM_DATAPOINTS][2];
char map[MAP_SIZE][MAP_SIZE] = {};

/*
 * This function outputs the result. 
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

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
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
static void generate_test(unsigned int rocks_pos[][2], double rocks_vel[][2], double datapoints[][2], unsigned int seed)
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.

    for (int i = 0; i < ROCKS_NUM; i++)
    {
        rocks_pos[i][0] = generator() % MAP_SIZE;
        rocks_pos[i][1] = generator() % MAP_SIZE;
        rocks_vel[i][0] = MAX_VELO * double(generator()) / double(generator.max());
        rocks_vel[i][1] = MAX_VELO * double(generator()) / double(generator.max());
        if (i < NUM_DATAPOINTS)
        {
            datapoints[i][0] = MAX_X * double(generator()) / double(generator.max());
            datapoints[i][1] = MAX_X * double(generator()) / double(generator.max());
        }
    }
}

/*
 * This function outputs the decryptedMessage. 
 */
static void outputResult(unsigned int crashed_count)
{

    std::cout << "Total crashed count: " << crashed_count << std::endl
              << "DONE" << std::endl;
}

#endif // UTILITY_H
