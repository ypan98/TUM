//
// Created by Vincent Bode on 02/05/2020.
//

#ifndef ASSIGNMENTS_VIDEOOUTPUT_H
#define ASSIGNMENTS_VIDEOOUTPUT_H

#include "a-star-navigator.h"
#include <vector>

#define NUM_CHANNELS 3

class VideoOutput {
    constexpr static unsigned char searchSpaceColor[3] = {255, 0, 0};
    constexpr static unsigned char pathColor[3] = {0,255, 0};

public:
    static void writeVideoFrames(std::vector<Position2D> &path, ProblemData &problemData);

    static void loadTextures();

    static void loadTexture(const char *filename, unsigned char target[MAP_SIZE][MAP_SIZE][3]);

    static void beginVideoOutput();

    static void endVideoOutput();

};


#endif //ASSIGNMENTS_VIDEOOUTPUT_H
