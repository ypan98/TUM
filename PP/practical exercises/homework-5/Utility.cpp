

#include <algorithm>
#include <getopt.h>
#include "Utility.h"
#include "lib/PerlinNoise.hpp"
#include "a-star-navigator.h"


void Utility::writeMap(unsigned int seed, ProblemData &problemData) {
    const siv::PerlinNoise perlinNoise(seed);
#pragma omp parallel for schedule(static, 32)
    for (int x = 0; x < MAP_SIZE; x++) {
        for (int y = 0; y < MAP_SIZE; y++) {
            problemData.islandMap[x][y] = perlinNoise.accumulatedOctaveNoise2D_0_1(x * FREQUENCY_WIND / MAP_SIZE,
                                                                                   y * FREQUENCY_WIND / MAP_SIZE,
                                                                                   OCTAVES_WAVES);
//            islandMap[x][y] = 0.9f;
        }
    }
}

void Utility::parse_input(bool& outputVisualization, bool& constructPathForVisualization, int& numProblems, int& option, int argc, char** argv) {
    while ((option = getopt(argc, argv, "vphn:")) != -1) {
        switch (option) {
        case 'v':
            outputVisualization = true;
            break;
        case 'p':
            constructPathForVisualization = true;
            break;
        case 'n':
            numProblems = strtol(optarg, nullptr, 0);
            if (numProblems <= 0) {
                std::cerr << "Error parsing number problems." << std::endl;
                exit(-1);
            }
            break;
        case 'h':
            std::cerr << "Usage: " << argv[0] << " [-v] [-p] [-n <numProblems>] [-h]" << std::endl
                << "-v: Output a visualization to file out.mp4. Requires FFMPEG to be in your $PATH to work."
                << std::endl
                << "-p: Also output the actual path used to reach Port Royal to the visualization. Can be slow"
                " and uses lots of memory." << std::endl
                << "-n: The number of problems to solve." << std::endl
                << "-h: Show this help topic." << std::endl;
            exit(-1);
        default:
            std::cerr << "Unknown option: " << (unsigned char)option << std::endl;
            exit(-1);
        }
    }

    std::cerr << "Solving " << numProblems <<
        " problems (visualization: " << outputVisualization << ", path visualization "
        << constructPathForVisualization << ")" << std::endl;

}

void Utility::writeInitialStormData(unsigned int seed, ProblemData &problemData) {
    unsigned int seeds[] = {seed * 2, seed / 2};
    const siv::PerlinNoise perlinNoise(seeds[0]);
#pragma omp parallel for schedule(static, 32)
    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            if (problemData.islandMap[x][y] >= LAND_THRESHOLD) {
                problemData.waveIntensity[0][x][y] = 0.0f;
                problemData.waveIntensity[1][x][y] = 0.0f;
            } else {
                // Make sure the gradient near coasts is not too large (don't create tsunamies).
                problemData.waveIntensity[0][x][y] =
                        perlinNoise.accumulatedOctaveNoise2D_0_1(x * FREQUENCY_WAVES / MAP_SIZE,
                                                                 y * FREQUENCY_WAVES / MAP_SIZE, OCTAVES_WAVES)
                        * std::clamp(4.0f * (LAND_THRESHOLD - problemData.islandMap[x][y]), 0.0f, 1.0f);
                problemData.waveIntensity[1][x][y] = problemData.waveIntensity[0][x][y];
            }
        }
    }
}

void Utility::generateProblem(unsigned int seed, ProblemData &problemData) {
    std::cerr << "Using seed " << seed << std::endl;
    if (seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    // Set the pseudo random number generator seed used for generating encryption keys
    srand(seed);

    writeMap(seed, problemData);
    writeInitialStormData(seed, problemData);

    // Ensure that the destination doesn't go out of bounds and that the ship doesn't start on land
    do {
        problemData.shipOrigin = Position2D(DISTANCE_TO_PORT + (rand() % (MAP_SIZE - 2 * DISTANCE_TO_PORT)),
                                            DISTANCE_TO_PORT + (rand() % (MAP_SIZE - 2 * DISTANCE_TO_PORT)));
    } while (problemData.islandMap[problemData.shipOrigin.x][problemData.shipOrigin.y] >= LAND_THRESHOLD);

    // Same for Port Royal
    do {
        int horizontalDistance = rand() % DISTANCE_TO_PORT;
        int verticalDistance = DISTANCE_TO_PORT - horizontalDistance;
        if (rand() % 2 == 1) { horizontalDistance = -horizontalDistance; }
        if (rand() % 2 == 1) { verticalDistance = -verticalDistance; }
        problemData.portRoyal = problemData.shipOrigin + Position2D(horizontalDistance, verticalDistance);
    } while (problemData.islandMap[problemData.portRoyal.x][problemData.portRoyal.y] >= LAND_THRESHOLD);

}

unsigned int Utility::readInput() {
    unsigned int seed = 0;
    std::cout << "READY" << std::endl;
    std::cin >> seed;

    return seed;
}

void Utility::writeOutput(int pathLength) {
    if (pathLength == -1) {
        std::cout << "Path length: no solution." << std::endl;
    } else {
        std::cout << "Path length: " << pathLength << std::endl;
    }
}

void Utility::stopTimer() {
    std::cout << "DONE" << std::endl;
}