//
// Created by Vincent Bode on 02/05/2020.
//

#ifndef ASSIGNMENTS_VIDEOOUTPUT_H
#define ASSIGNMENTS_VIDEOOUTPUT_H

#include "life.h"
#include <vector>
#include <cstdio>

#define NUM_CHANNELS 3

class VideoOutput {
    constexpr static unsigned char aliveColor[3] = { 255, 255, 255 };
    constexpr static unsigned char deadColor[3] = { 0, 0, 0 };

public:
    static void writeVideoFrames(ProblemData& problemData);

    static void beginVideoOutput();

    static void endVideoOutput();

    /**
     * Write output to a png file. If filename is not specified, a default file name is chosen.
     * @param grid
     * @param filename
     */
    static void saveToFile(Grid& grid, char const* filename);

    static void printGrid(FILE* file, Grid& grid);

    static void prepareFrame(Grid& grid);
};


#endif //ASSIGNMENTS_VIDEOOUTPUT_H
