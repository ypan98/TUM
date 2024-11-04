#include "Utility.h"
#include <iostream>

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1

float update_block(int st_row, int end_row, int st_col, int end_col, float *old_wave_map, float *new_wave_map)
{
    float highest = 0.0f;
    for (int row = st_row; row < end_row; row++)
    {
        for (int col = st_col; col < end_col; col++)
        {
            // wrap the map (for example, the left column of the leftmost column is the rightmost column)
            int upper_row = (row - 1 + MAP_HEIGHT) % MAP_HEIGHT;
            int lower_row = (row + 1 + MAP_HEIGHT) % MAP_HEIGHT;
            int left_col = (col - 1 + MAP_WIDTH) % MAP_WIDTH;
            int right_col = (col + 1 + MAP_WIDTH) % MAP_WIDTH;

            // average the 4 neighbors and assign to the current position
            new_wave_map[row * MAP_WIDTH + col] = 0.2 * (old_wave_map[upper_row * MAP_WIDTH + col] + old_wave_map[lower_row * MAP_WIDTH + col] + old_wave_map[row * MAP_WIDTH + left_col] + old_wave_map[row * MAP_WIDTH + right_col] + old_wave_map[row * MAP_WIDTH + col]);
            if (new_wave_map[row * MAP_WIDTH + col] > highest){
                highest = new_wave_map[row * MAP_WIDTH + col];
            }
        }
    }
    return highest;
}

int main()
{
    unsigned int seed = readInput();

#ifdef PRINT_TIME
    TicToc total_time;
#endif
    float *old_wave_map = new float [MAP_HEIGHT * MAP_WIDTH];
    float *new_wave_map = new float [MAP_HEIGHT * MAP_WIDTH];
    float *highest_result = new float[ITER_NUM];
    generate_test(seed, old_wave_map);

    for (int i = 0; i < ITER_NUM; i++)
    {
        highest_result[i] = update_block(0, MAP_HEIGHT, 0, MAP_WIDTH, old_wave_map, new_wave_map);
        swap_ptr(old_wave_map, new_wave_map);
    }

    outputResult(highest_result);
#ifdef PRINT_TIME
    std::cerr << "time used: " << total_time.toc() << "ms.\n";
#endif
    return 0;
}
