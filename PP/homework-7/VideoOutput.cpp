//
// Created by Vincent Bode on 02/05/2020.
//

#include "VideoOutput.h"
#include "life.h"
#include "lib/TinyPngOut.hpp"

#include <iostream>
#include <vector>
#include <functional>
#include <cstring>
#include <fstream>

// Initialize to zero
unsigned char frame[GRID_SIZE][GRID_SIZE][NUM_CHANNELS] = { {} };

FILE* ffmpeg;

void VideoOutput::beginVideoOutput() {
    std::cerr << "Launching ffmpeg... (writing to out.mp4)" << std::endl;
    ffmpeg = popen(
        ("ffmpeg -loglevel warning -f rawvideo -s "
            + std::to_string(GRID_SIZE) + "x" + std::to_string(GRID_SIZE) +
            " -r 12 -pix_fmt rgb24 -i - -y -pix_fmt yuv420p -c:v libx264 -preset veryfast -sn out.mp4").c_str(),
        "w");
    if (!ffmpeg) {
        std::cerr << "Could not launch ffmpeg to encode video output! " << std::endl;
        perror("Error launching ffmpeg");
        exit(-1);
    }
}

void VideoOutput::endVideoOutput() {
    std::cerr << "Waiting for ffmpeg to finish writing video..." << std::endl;
    int returnCode = pclose(ffmpeg);
    if (returnCode != 0) {
        std::cerr << "ffmpeg reported error code: " << returnCode << std::endl;
        exit(-1);
    }
}

void VideoOutput::writeVideoFrames(ProblemData& problemData) {
    prepareFrame(*problemData.readGrid);

    //    std::cerr << "Outputting video frame " << t << std::endl;
    fwrite(frame[0][0], sizeof(frame[0][0][0]), NUM_CHANNELS * GRID_SIZE * GRID_SIZE, ffmpeg);
}

void VideoOutput::prepareFrame(Grid& grid) {
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            memcpy(frame[x][y], grid[x][y] ? aliveColor : deadColor, sizeof(unsigned char) * NUM_CHANNELS);
        }
    }
}

void VideoOutput::saveToFile(Grid& grid, char const* filename) {
    /*
      If you do not want to specify a filename, just pass an empty string.
     */

    prepareFrame(grid);

    std::filebuf filebuf;

    if (filename == nullptr || strlen(filename) == 0) {
        filename = "./grid.png";
    }

    std::ofstream outputStream(filename, std::ios::binary);
    TinyPngOut out(GRID_SIZE, GRID_SIZE, outputStream);
    out.write(frame[0][0], GRID_SIZE * GRID_SIZE);
    outputStream.close();
}

void VideoOutput::printGrid(FILE* file, Grid& grid) {
    printf("\n");

    for (int i = 1; i < GRID_SIZE - 1; i++) {
        for (int j = 1; j < GRID_SIZE - 1; j++) {
            fprintf(file, "%d ", grid[i][j]);
        }
        printf("\n");
    }
}
