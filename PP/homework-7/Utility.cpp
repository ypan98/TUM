#include <iostream>
#include <random>
#include "Utility.h"


std::minstd_rand randomEngine;
uint_fast32_t cachedValue;
uint_fast32_t bitMask = 0;

/**
 * Extract random bits from a C++ generator.
 * @return
 */
inline bool generateBit() {
    if (!bitMask) {
        cachedValue = randomEngine();
        bitMask = 1;
    }
    bool value = cachedValue & bitMask;
    bitMask = bitMask << 1;
    return value;
}

void seedGenerator(unsigned long long seed) {
    randomEngine = std::minstd_rand(seed);
}

/**
 * Generates the problem by getting a random seed from stdin and filling the grid with it.
 * @param data
 */
void Utility::readProblemFromInput(ProblemData& data) {
    auto& grid = *data.readGrid;

    unsigned int seed = 0;
    std::cout << "READY" << std::endl;
    std::cin >> seed;

    std::cout << "Using seed " << seed << std::endl;
    if (seed == 0) {
        std::cout << "Warning: default value 0 used as seed." << std::endl;
    }

    // "random" numbers
    seedGenerator(seed);

    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i += 1) {
        *(grid[0] + i) = generateBit();
    }
}

/**
 * Counts the number of cells that are alive on the entire read grid, except for the outer cells used to make the grid
 * periodic. Copy this function into your source code and modify it if you need to.
 * @param data The problem data, from which the read grid is counted.
 * @return The number of alive cells in the grid.
 */
int Utility::countAlive(ProblemData& data) {
    auto& grid = *data.readGrid;
    int counter = 0;
    for (int x = 1; x < GRID_SIZE - 1; x++) {
        for (int y = 1; y < GRID_SIZE - 1; y++) {
            if (grid[x][y]) {
                counter++;
            }
        }
    }
    return counter;
}

/**
 * Counts the number of alive cells on the read grid. Depending on how you implement your parallelization, you might
 * need to modify this function. If you need to do so, copy it to your submission file and edit it.
 * @param iteration The current iteration
 * @param data The read grid to count the number of alive cells from.
 */
void Utility::outputIntermediateSolution(int iteration, ProblemData& data) {
    std::cout << "Iteration " << iteration << ": " << countAlive(data) << " cells alive." << std::endl;
}

/**
 * The same as outputIntermediateSolution, but also prints done.
 * @param data
 */
void Utility::outputSolution(ProblemData& data) {
    outputIntermediateSolution(NUM_SIMULATION_STEPS, data);
    std::cout << "DONE" << std::endl;
}

ComparisonResult Utility::compareGrids(Grid& grid_seq, Grid& grid_par)
{
    /*
      Compare the inner parts of grid_seq with grid_par i.e. excluding the padding rows. Writes the index of first
      difference in "row" and "col".

      Returns
      0: if arrays are equal
      1: if arrays are different
     */

    ComparisonResult result{};

    for (int i = 1; i < GRID_SIZE - 1; i++)
    {
        for (int j = 1; j < GRID_SIZE - 1; j++)
        {
            if (grid_seq[i][j] != grid_par[i][j])
            {
                result.firstDifferenceRow = i;
                result.firstDifferenceColumn = j;
                result.gridsEqual = false;
                return result;
            }
        }
    }
    result.firstDifferenceRow = -1;
    result.firstDifferenceColumn = -1;
    result.gridsEqual = true;
    return result;
}
