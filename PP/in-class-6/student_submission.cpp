#include "Utility.h"
#include <iostream>
#include <mpi.h>

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1

/**
 * This is a function that updates a block of the wave map specified by the starting and ending rows and columns
 * You don't need to modify this function. But it might be helpful to understand the update process.
 */
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


int main(int argc, char **argv)
{
    // TODO: initialize the MPI process
    MPI_Init(&argc, &argv);

    // define the rank and size variable
    int rank, size;

    // TODO: get the process rank from MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // TODO: get the total processes size from MPI

    float *old_wave_map = new float [MAP_HEIGHT * MAP_WIDTH];
    float *new_wave_map = new float [MAP_HEIGHT * MAP_WIDTH];
    float *highest_result = new float[ITER_NUM];

    if (rank == 0){
        unsigned int seed = readInput();
        generate_test(seed, old_wave_map);
    }
    // pass the wave map to all other processes
    MPI_Bcast(old_wave_map, MAP_HEIGHT * MAP_WIDTH, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef PRINT_TIME
    TicToc total_time;
#endif

    // the number of rows calculated by a process
    int proc_rows = MAP_HEIGHT / size;

    // the starting row in this process
    int st_row = rank * proc_rows;

    // the end row(not included) in this process
    int end_row = (rank + 1) * proc_rows;

    for (int i = 0; i < ITER_NUM; i++)
    {
        // communicate the ghost layers before update
        if (rank == 0)
        {
            // send lowest row in block to process with rank 1
            MPI_Send(old_wave_map + (end_row - 1) * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

            // receive upper ghost layer from process with rank size-1
            MPI_Recv(old_wave_map + (MAP_HEIGHT - 1) * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, size - 1, 0, MPI_COMM_WORLD, nullptr);

            // send uppermost row in block to process with rank size-1
            MPI_Send(old_wave_map + st_row * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, size - 1, 1, MPI_COMM_WORLD);

            // receive lower ghost layer from process with rank 1
            MPI_Recv(old_wave_map + end_row * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, nullptr);
        }
        else
        {
            /*
             * TODO: use four lines of codes to finish the following receive and send process for rank other than 0
             *       pay attention to the order of the send and receive.
             */

            // TODO: receive upper ghost layer from process with rank rank-1
            MPI_Recv(old_wave_map + (st_row - 1) * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, nullptr);

            // TODO: send lowest row in block to the process with rank (rank+1)%size
            MPI_Send(old_wave_map + (end_row - 1) * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, (rank + 1) % size, 0, MPI_COMM_WORLD);

            // TODO: receive lower ghost layer from process with rank (rank+1)%size
            MPI_Recv(old_wave_map + (end_row % MAP_HEIGHT) * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, (rank + 1) % size, 1, MPI_COMM_WORLD, nullptr);

            // TODO: send uppermost row in block to the process with rank rank-1
            MPI_Send(old_wave_map + st_row * MAP_WIDTH, MAP_WIDTH, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);

        }
        highest_result[i] = update_block(st_row, end_row, 0, MAP_WIDTH, old_wave_map, new_wave_map);
        swap_ptr(old_wave_map, new_wave_map);

        float highest = 0;
        MPI_Reduce(highest_result + i, &highest, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            highest_result[i] = highest;
        }
    }

    if (rank == 0)
    {
        outputResult(highest_result);
    }

    // TODO: finalize the MPI process
    MPI_Finalize();

#ifdef PRINT_TIME
    if (rank == 0)
    {
        std::cerr << "time used: " << total_time.toc() << "ms.\n";
    }
#endif
    return 0;
}
