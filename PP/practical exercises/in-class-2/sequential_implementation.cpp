#include "Utility.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cmath>

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1

unsigned int calc_hits(unsigned int ship_r, unsigned int ship_c)
{
    unsigned int count = 0;
    for (int i = 0; i < ROCKS_NUM; i++)
    {
        auto &row = rocks_pos[i][0];
        auto &col = rocks_pos[i][1];
        if (row == ship_r && col == ship_c)
        {
            count++;
        }
    }
    return count;
}

void compute_vel(unsigned int pos, double current_vel, double *target_vel)
{
    double x = std::fmod(std::pow(current_vel, pos % NUM_DATAPOINTS), MAX_X);
    /*
     * You do not have to modify or understand this code.
     * But in your free time, if you manage to find an optimization to this algorithm (maybe another algorithm?), 
     * you will get our personal appreciation.
     * This algorithm is not relevant for the exam and is not part of the course material. 
     */
    double p[NUM_DATAPOINTS][NUM_DATAPOINTS];

    for(int i = 0; i < NUM_DATAPOINTS; i++)
    {
        p[i][0] =  datapoints[i][1];
    }

    for(int k = 1; k < NUM_DATAPOINTS; k++)
    {
        for(int i = 0; i < NUM_DATAPOINTS - k; i++)
        {
            p[i][k] = p[i][k - 1] + ((x - datapoints[i][0])/(datapoints[i + k][0] - datapoints[i][0])) * (p[i + 1][k - 1] - p[i][k - 1]);
        }
    }

    *target_vel = std::fmod(p[0][NUM_DATAPOINTS - 1], MAX_VELO);
}

void update_rock(int rock_id)
{
    unsigned int &rock_r = rocks_pos[rock_id][0];
    unsigned int &rock_c = rocks_pos[rock_id][1];
    double &rock_vr = rocks_vel[rock_id][0];
    double &rock_vc = rocks_vel[rock_id][1];
    compute_vel(rock_r, rock_vr, &rock_vr);
    compute_vel(rock_c, rock_vc, &rock_vc);

    double tmp = rock_r + rock_vr;
    rock_r = (unsigned int)((long)tmp % MAP_SIZE);

    tmp = rock_c + rock_vc;
    rock_c = (unsigned int)((long)tmp % MAP_SIZE);
}

int main()
{
    unsigned int seed = readInput();
#ifdef PRINT_TIME
    TicToc total_time;
#endif
    generate_test(rocks_pos, rocks_vel, datapoints, seed);
    unsigned int crashed_count = 0;

    for (unsigned int i = 0; i < MAP_SIZE; i++)
    {
        for (int k = 0; k < ROCKS_NUM; k++)
        {
            update_rock(k);
        }
        crashed_count += calc_hits(i, i);
    }
    outputResult(crashed_count);
#ifdef PRINT_TIME
    std::cerr << "time used: " << total_time.toc() << "ms.\n";
#endif
}