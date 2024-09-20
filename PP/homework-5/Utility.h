#ifndef ASSIGNMENTS_UTILITY_H
#define ASSIGNMENTS_UTILITY_H

#include "a-star-navigator.h"
#include <vector>

static Position2D neighbours[] = { Position2D(-1, 0),Position2D(0, -1),Position2D(1, 0),Position2D(0, 1),
Position2D(-1, -1),Position2D(-1, 1),Position2D(1, -1),Position2D(1, 1),Position2D(0, 0) };

class Utility {
public:
    static void parse_input(bool& outputVisualization, bool& constructPathForVisualization, int& numProblems, int& option, int argc, char** argv);

    static void writeMap(unsigned int seed, ProblemData& problemData);

    static void writeInitialStormData(unsigned int seed, ProblemData& problemData);

    static void generateProblem(unsigned int seed, ProblemData &problemData);

    static unsigned int readInput();

    // Output the solution to a problem
    static void writeOutput(int pathLength);

    // This will stop the timer. No more output will be accepted after this call.
    static void stopTimer();
};


#endif //ASSIGNMENTS_UTILITY_H
