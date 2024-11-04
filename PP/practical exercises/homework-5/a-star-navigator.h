#ifndef ASSIGNMENTS_A_STAR_NAVIGATOR_H
#define ASSIGNMENTS_A_STAR_NAVIGATOR_H

#define TIME_STEPS 640
#define MAP_SIZE 1024
#define LAND_THRESHOLD 0.65f

#define ATTACK_FACTOR 0.5f
#define ENERGY_PRESERVATION_FACTOR 20.0f

#define OCTAVES_WAVES 4
#define FREQUENCY_WAVES 30.0f

#define OCTAVES_WIND 4
#define FREQUENCY_WIND 4.0f

#define DISTANCE_TO_PORT 420
#define SHIP_THRESHOLD 0.15f

#define VIS_TIMES 30
#define INT_LIM 4294967295
#define JUMP_SIZE 2654435769


#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_map>

/*#define TIME(y, x) {     \
//    auto startTime = std::chrono::high_resolution_clock::now(); \
//    y;\
//    auto stopTime = std::chrono::high_resolution_clock::now(); \
//    std::cerr << "Timed function " << x << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count() << "ms." << std::endl; \
}*/
#define TIME(y, x) { y; }

class Position2D {
public:
    int x;
    int y;

    Position2D(int x, int y) : x{x}, y{y} {}

    Position2D() : Position2D(-1, -1) {}

    Position2D operator+(const Position2D &p2) const {
        return {x + p2.x, y + p2.y};
    }

    Position2D operator-(const Position2D &p2) const {
        return {x - p2.x, y - p2.y};
    }

    bool operator==(const Position2D &p2) const {
        return x == p2.x && y == p2.y;
    }

    bool operator!=(const Position2D &p2) const {
        return !(*this == p2);
    }

    int distanceTo(const Position2D &other) const {
        return abs(this->x - other.x) + abs(this->y - other.y);
    }
};


class Position2DTime {
public:
    int time;
    Position2D position;

    // Constructor assigning values.
    Position2DTime(int time, Position2D position) : time{time}, position{position} {}

    Position2DTime(int time, int x, int y) : Position2DTime(time, Position2D{x, y}) {}

    Position2DTime() : Position2DTime(0, 0, 0) {}

    bool operator==(const Position2DTime &other) const {
        return time == other.time && position == other.position;
    }

    bool operator!=(const Position2DTime &other) const {
        return !(*this == other);
    }

};

/*
 * Sorts the 2d position with times according to how promising they are to reach the destination first. The indicator
 * used is the distance already travelled + the remaining direct distance to destination.
 */
class Position2DTimeOrdering {
    Position2D portRoyal;
public:
    explicit Position2DTimeOrdering(Position2D &portRoyal) : portRoyal{portRoyal} {}

    bool operator()(const Position2DTime &self, const Position2DTime &other) {
        return self.time + self.position.distanceTo(portRoyal) > other.time + other.position.distanceTo(portRoyal);
    }
};

class Position2DTimeHash {
public:
    std::size_t operator()(const Position2DTime &position2DTime) const {
        return + std::hash<int>{}(position2DTime.time)
               + std::hash<int>{}(position2DTime.position.x)
               + std::hash<int>{}(position2DTime.position.y);
    }
};

class Position2DHash {
public:
    std::size_t operator()(const Position2D &position2D) const {
        return + std::hash<int>{}(position2D.x)
               + std::hash<int>{}(position2D.y);
    }
};

/*
class Position2DTimeCompare {
public:
    bool operator() (const Position2DTime &self, const Position2DTime &other) {
        return self.time == other.time && self.position == other.position;
    }
};
*/

class PositionInformation {
public:
    int distanceFromStart;
    Position2D previousPosition;

    PositionInformation(int distanceFromStart, Position2D previousPosition) :
            distanceFromStart{distanceFromStart},
            previousPosition{previousPosition} {}

    PositionInformation() : PositionInformation(std::numeric_limits<int>::max(),
                                                Position2D()) {}
};



class ProblemData {
public:
    Position2D shipOrigin;
    Position2D portRoyal;

    float islandMap[MAP_SIZE][MAP_SIZE];
    float waveIntensity[3][MAP_SIZE][MAP_SIZE];

    float (*currentWaveIntensity)[MAP_SIZE][MAP_SIZE];
    float (*lastWaveIntensity)[MAP_SIZE][MAP_SIZE];
    float (*secondLastWaveIntensity)[MAP_SIZE][MAP_SIZE];

    bool possibleShipPositionsA[MAP_SIZE][MAP_SIZE] = {{}};
    bool possibleShipPositionsB[MAP_SIZE][MAP_SIZE] = {{}};

    bool (*currentShipPositions)[MAP_SIZE][MAP_SIZE];
    bool (*previousShipPositions)[MAP_SIZE][MAP_SIZE];

    std::unordered_map<Position2D, Position2D, Position2DHash> nodePredecessors[TIME_STEPS];
    int numPredecessors = 0;

    bool outputVisualization = false;
    bool constructPathForVisualization = false;

    ProblemData() {
        currentShipPositions = &possibleShipPositionsA;
        previousShipPositions = &possibleShipPositionsB;

        secondLastWaveIntensity = &waveIntensity[0];
        lastWaveIntensity = &waveIntensity[1];
        currentWaveIntensity = &waveIntensity[2];
    }

    void flipWaveBuffers() {
        auto *tmp = secondLastWaveIntensity;
        secondLastWaveIntensity = lastWaveIntensity;
        lastWaveIntensity = currentWaveIntensity;
        currentWaveIntensity = tmp;
    }

    void flipSearchBuffers() {
        // Switch the current data with the previous data. The previous data will then be overwritten
        auto *tmp = currentShipPositions;
        currentShipPositions = previousShipPositions;
        previousShipPositions = tmp;

    }
};

#endif //ASSIGNMENTS_A_STAR_NAVIGATOR_H