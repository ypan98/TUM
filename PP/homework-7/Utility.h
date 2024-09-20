#ifndef _HELPER_H_LIFE
#define _HELPER_H_LIFE

#include "life.h"

class ComparisonResult {
public:
    bool gridsEqual;
    int firstDifferenceRow;
    int firstDifferenceColumn;
};

class Utility {
public:
    /**
     * Counts the number of cells that are alive on the entire read grid, except for the outer cells used to make the grid
     * periodic. Copy this function into your source code and modify it if you need to.
     * @param data The problem data, from which the read grid is counted.
     * @return The number of alive cells in the grid.
     */
    static int countAlive(ProblemData& data);

    /**
     * Generates the problem by getting a random seed from stdin and filling the grid with it.
     * @param data
     */
    static void readProblemFromInput(ProblemData& data);

    /**
     * Counts the number of alive cells on the read grid. Depending on how you implement your parallelization, you might
     * need to modify this function. If you need to do so, copy it to your submission file and edit it.
     * @param iteration The current iteration
     * @param data The read grid to count the number of alive cells from.
     */
    static void outputSolution(ProblemData& data);

    /**
     * The same as outputIntermediateSolution, but also prints done.
     * @param data
     */
    static void outputIntermediateSolution(int iteration, ProblemData& data);

    /**
     * Compares two grid to check whether they are equal. If they aren't, the row and column of the first mismatch
     * are returned.
     * @param grid_seq Grid 1
     * @param grid_par Grid 2
     * @return Result of the comparison
     */
    static ComparisonResult compareGrids(Grid& grid_seq, Grid& grid_par);
};

#endif
