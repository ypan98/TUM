//
// Created by Dennis-Florian Herr on 13/06/2022.
//

#include <string>
#include <deque>
#include <future>
#include <functional>

#include "Utility.h"

#define MEASURE_TIME false

struct Problem {
    Sha1Hash sha1_hash;
    int problemNum;
};


class ProblemQueue {
    public:
        void push(Problem problem){
            problemQueue.push_back(problem);
        }

        Problem pop(){
            Problem p = problemQueue.front();
            problemQueue.pop_front();
            return p;
        }

        bool empty(){
            return problemQueue.empty();
        }

    private:
        std::deque<Problem> problemQueue;

};

ProblemQueue problemQueue;


// generate numProblems sha1 hashes with leadingZerosProblem leading zero bits
// This method is intentionally compute intense so you can already start working on solving
// problems while more problems are generated
void generateProblem(int seed, int numProblems, int leadingZerosProblem){
    srand(seed+1);

    for(int i = 0; i < numProblems; i++){
        std::string base = std::to_string(rand()) + std::to_string(rand());
        Sha1Hash hash = Utility::sha1(base);
        do{
            // we keep hashing ourself until we find the desired amount of leading zeros
            hash = Utility::sha1(hash);
        }while(Utility::count_leading_zero_bits(hash) < leadingZerosProblem);
        problemQueue.push(Problem{hash, i});
    }
}

// This method repeatedly hashes itself until the required amount of leading zero bits is found
Sha1Hash findSolutionHash(Sha1Hash hash, int leadingZerosSolution){
    do{
        // we keep hashing ourself until we find the desired amount of leading zeros
        hash = Utility::sha1(hash);
    }while(Utility::count_leading_zero_bits(hash) < leadingZerosSolution);

    return hash;
}

int main(int argc, char *argv[]) {
    int leadingZerosProblem = 8;
    int leadingZerosSolution = 11;
    int numProblems = 10000;

    //Not interesting for parallelization
    Utility::parse_input(numProblems, leadingZerosProblem, leadingZerosSolution, argc, argv);
    Sha1Hash solutionHashes[numProblems];

    unsigned int seed = Utility::readInput();

    #if MEASURE_TIME
    struct timespec generation_start, generation_end;
    clock_gettime(CLOCK_MONOTONIC, &generation_start);
    #endif

    generateProblem(seed, numProblems, leadingZerosProblem);

    #if MEASURE_TIME
    clock_gettime(CLOCK_MONOTONIC, &generation_end);
    double generation_time = (((double) generation_end.tv_sec + 1.0e-9 * generation_end.tv_nsec) - ((double) generation_start.tv_sec + 1.0e-9 * generation_start.tv_nsec));
    fprintf(stderr, "Generate Problem time:  %.7gs\n", generation_time);

    struct timespec solve_start, solve_end;
    clock_gettime(CLOCK_MONOTONIC, &solve_start);
    #endif

    while(!problemQueue.empty()) {
        Problem p = problemQueue.pop();
        solutionHashes[p.problemNum] = findSolutionHash(p.sha1_hash, leadingZerosSolution);
    }

    #if MEASURE_TIME
    clock_gettime(CLOCK_MONOTONIC, &solve_end);
    double solve_time = (((double) solve_end.tv_sec + 1.0e-9 * solve_end.tv_nsec) - ((double) solve_start.tv_sec + 1.0e-9 * solve_start.tv_nsec));
    fprintf(stderr, "Solve Problem time:     %.7gs\n", solve_time);
    #endif


    Sha1Hash solution;
    // guarantee initial solution hash data is zero
    memset(solution.data, 0,  SHA1_SIZE);
    // this doesn't need parallelization. it's neglectibly fast
    for(int i = 0; i < numProblems; i++){
        solution = Utility::sha1(solution, solutionHashes[i]);
    }

    Utility::printHash(solution);
    printf("DONE\n");

    return 0;
}
