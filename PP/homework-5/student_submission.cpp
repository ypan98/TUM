#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <getopt.h>
#include <cstring>

#include "a-star-navigator.h"
#include "VideoOutput.h"
#include "Utility.h"

void simulate_waves(ProblemData &problemData) {
    auto &islandMap = problemData.islandMap;
    float (&secondLastWaveIntensity)[MAP_SIZE][MAP_SIZE] = *problemData.secondLastWaveIntensity;
    float (&lastWaveIntensity)[MAP_SIZE][MAP_SIZE] = *problemData.lastWaveIntensity;
    float (&currentWaveIntensity)[MAP_SIZE][MAP_SIZE] = *problemData.currentWaveIntensity;

    #pragma omp parallel for schedule(dynamic)
    for (int x = 1; x < MAP_SIZE - 1; ++x) {
        for (int y = 1; y < MAP_SIZE - 1; ++y) {

            // Simulate some waves

            // The acceleration is the relative difference between the current point and the last.
            float acceleration = lastWaveIntensity[x][y - 1]
                                 + lastWaveIntensity[x - 1][y]
                                 + lastWaveIntensity[x + 1][y]
                                 + lastWaveIntensity[x][y + 1]
                                 - 4 * lastWaveIntensity[x][y];

            // The acceleration is multiplied with an attack value, specifying how fast the system can accelerate.
            acceleration *= ATTACK_FACTOR;

            // The last_velocity is calculated from the difference between the last intensity and the
            // second to last intensity
            float last_velocity = lastWaveIntensity[x][y] - secondLastWaveIntensity[x][y];

            // energy preserved takes into account that storms lose energy to their environments over time. The
            // ratio of energy preserved is higher on open water, lower close to the shore and 0 on land.
            float energyPreserved = std::clamp(
                    ENERGY_PRESERVATION_FACTOR * (LAND_THRESHOLD - 0.15f * islandMap[x][y]), 0.0f, 1.0f);

            // There aren't any waves on land.
            if (islandMap[x][y] >= LAND_THRESHOLD) {
                currentWaveIntensity[x][y] = 0.0f;
            } else {
                currentWaveIntensity[x][y] =
                        std::clamp(lastWaveIntensity[x][y] + (last_velocity + acceleration) * energyPreserved, 0.0f, 1.0f);
            }
        }
    }
}


 // Since all pirates like navigating by the stars, Captain Jack's favorite pathfinding algorithm is called A*.
 // Unfortunately, sometimes you just have to make do with what you have. So here we use a search algorithm that searches
 // the entire domain every time step and calculates all possible ship positions.
bool findPathWithExhaustiveSearch(ProblemData &problemData, int timestep,
                                  std::vector<Position2D> &pathOutput) {
    auto &start = problemData.shipOrigin;
    auto &portRoyal = problemData.portRoyal;
    auto &islandMap = problemData.islandMap;
    auto &currentWaveIntensity = *problemData.currentWaveIntensity;

    int numPossiblePositions = 0;

    bool (&currentShipPositions)[MAP_SIZE][MAP_SIZE] = *problemData.currentShipPositions;
    bool (&previousShipPositions)[MAP_SIZE][MAP_SIZE] = *problemData.previousShipPositions;

    // We could always have been at the start in the previous frame since we get to choose when we start our journey.
    previousShipPositions[start.x][start.y] = true;

    // Ensure that our new buffer is set to zero. We need to ensure this because we are reusing previously used buffers.
    memset(currentShipPositions, 0, sizeof(currentShipPositions));
    volatile bool found = false;

    // Do the actual path finding.
    #pragma omp parallel for schedule(dynamic) reduction(+: numPossiblePositions) shared(found)
    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            if (found) continue;
            // If there is no possibility to reach this position then we don't need to process it.
            if (!previousShipPositions[x][y]) {
                continue;
            }
            Position2D previousPosition(x, y);

            // The Jolly Mon (Jack's ship) is not very versatile. It can only move along the four cardinal directions by one
            // square each and along their diagonals. Alternatively, it can just try to stay where it is.
            // If we are not yet done then we have to take a look at our neighbors.
            for (Position2D &neighbor: neighbours) {
                // Get the neighboring position we are examining. It is one step further in time since we have to move
                // there first.
                Position2D neighborPosition = previousPosition + neighbor;

                // If position is out of bounds, skip it
                if (neighborPosition.x < 0 || neighborPosition.y < 0
                    || neighborPosition.x >= MAP_SIZE || neighborPosition.y >= MAP_SIZE) {
                    continue;
                }

                // If this position is already marked, skip it
                bool current_pos_marked;
                #pragma omp atomic read
                current_pos_marked = currentShipPositions[neighborPosition.x][neighborPosition.y];
                if (current_pos_marked)  continue;
                

                // If we can't sail to this position because it is either on land or because the wave height is too
                // great for the Jolly Mon to handle, skip it
                if (islandMap[neighborPosition.x][neighborPosition.y] >= LAND_THRESHOLD ||
                    currentWaveIntensity[neighborPosition.x][neighborPosition.y] >= SHIP_THRESHOLD) {
                    continue;
                }

                if (problemData.constructPathForVisualization) {
                    // Add the previous node as the method we used to get here. This is only needed to draw the path for
                    // the output visualization.
                    if (neighborPosition.distanceTo(portRoyal) <= TIME_STEPS - timestep) {
                        Position2D &predecessor = problemData.nodePredecessors[timestep][neighborPosition];
                        predecessor = previousPosition;
                    }
                }

                // If we reach Port Royal, we win.
                if (neighborPosition == portRoyal) {
                    if (problemData.outputVisualization) {
                        // We flip the search buffer back to the previous one to prevent drawing a half finished buffer
                        // to screen (purely aesthetic).
                        problemData.flipSearchBuffers();
                    }
                    if (problemData.constructPathForVisualization) {
                        try {
                            // Trace back our path from the end to the beginning. This is just used to draw a path into
                            // the output video
                            Position2D pathTraceback = neighborPosition;
                            pathOutput.resize(timestep + 1);
                            int tracebackTimestep = timestep;
                            while (pathTraceback != start) {
                                if (tracebackTimestep <= 0) {
                                    std::cerr << "Traceback did not lead back to origin before timestep 0!"
                                              << std::endl;
                                    break;
                                }
                                pathOutput[tracebackTimestep] = pathTraceback;
                                pathTraceback = problemData.nodePredecessors[tracebackTimestep].at(pathTraceback);
                                tracebackTimestep--;
                            }
                        } catch (std::out_of_range& e) {
                            std::cerr << "Path traceback out of range: " << e.what() << std::endl;
                        }
                    }
                    found = true;
                }
                #pragma omp atomic write
                currentShipPositions[neighborPosition.x][neighborPosition.y] = true;
                numPossiblePositions++;
            }
        }
    }
    if (found) return true;
    // This is not strictly required but can be used to track how much additional memory our path traceback structures
    // are using.
    problemData.numPredecessors += problemData.nodePredecessors[timestep].size();

    return false;
}


 // Your main simulation routine.
int main(int argc, char *argv[]) {
    bool outputVisualization = false;
    bool constructPathForVisualization = false;
    int numProblems = 1;
    int option;

    //Not interesting for parallelization
    Utility::parse_input(outputVisualization, constructPathForVisualization, numProblems, option, argc, argv);
    
    // Fetch the seed from our container host used to generate the problem. This starts the timer.
    unsigned int seed = Utility::readInput();

    if (outputVisualization) {
        VideoOutput::beginVideoOutput();
    }

    // Note that on the submission server, we are solving "numProblems" problems
    for (int problem = 0; problem < numProblems; ++problem) {
        auto *problemData = new ProblemData();
        problemData->outputVisualization = outputVisualization;
        problemData->constructPathForVisualization = constructPathForVisualization;

        // Receive the problem from the system.
        Utility::generateProblem((seed + problem * JUMP_SIZE) & INT_LIM, *problemData);

        std::cerr << "Searching from ship position (" << problemData->shipOrigin.x << ", " << problemData->shipOrigin.y
                  << ") to Port Royal (" << problemData->portRoyal.x << ", " << problemData->portRoyal.y << ")."<< std::endl;

        int pathLength = -1;
        std::vector<Position2D> path;

        for (int t = 2; t < TIME_STEPS; t++) {
            // First simulate all cycles of the storm
            simulate_waves(*problemData);

            // Help captain Sparrow navigate the waves
            if (findPathWithExhaustiveSearch(*problemData, t, path)) {
                // The length of the path is one shorter than the time step because the first frame is not part of the
                // pathfinding, and the second frame is always the start position.
                pathLength = t - 1;
            }

            if (outputVisualization) {
                VideoOutput::writeVideoFrames(path, *problemData);
            }
            if (pathLength != -1) {
                break;
            }

            // Rotates the buffers, recycling no longer needed data buffers for writing new data.
            problemData->flipSearchBuffers();
            problemData->flipWaveBuffers();
        }
        // Submit our solution back to the system.
        Utility::writeOutput(pathLength);
        if (outputVisualization) {
            // Output the last frame some more times so that it's easier to see the path/result
            for (int i = 0; i < VIS_TIMES; ++i) {
                VideoOutput::writeVideoFrames(path, *problemData);
            }
        }

        delete problemData;
    }
    // This stops the timer by printing DONE.
    Utility::stopTimer();

    if (outputVisualization) {
        VideoOutput::endVideoOutput();
    }
    return 0;
}