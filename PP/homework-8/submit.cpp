//!submission_env OMP_PLACES=sockets
//!submission_env OMP_PROC_BIND=close
//!submission_env OMP_DISPLAY_ENV=verbose
//!submission_env OMP_DISPLAY_AFFINITY=true

//!mpirun_option --map-by socket:PE=4
//!mpirun_option --bind-to core


#include <cstddef> // for size_t
#include "Utility.h"
#include <algorithm>
#include "mpi.h"
#include <omp.h>
#include <chrono>

#define NUM_PROCESS_PER_ROW 2

double local_min{0.0};
double local_max{0.0};
int rank, size;

void compute_stencil(ProblemData &pd)
{
    // For evert cell with coordinates (y,x) compute the influx from neighbor cells
    // Apply reflecting boundary conditions
    double **domain = Utility::get_domain(pd);

    
    int rank_row_idx = rank / NUM_PROCESS_PER_ROW;
    int rank_col_idx = rank % NUM_PROCESS_PER_ROW;
    int half_dim = pd.dimension / 2;
    int end_row_idx = (rank_row_idx + 1) * half_dim;
    int end_col_idx = (rank_col_idx + 1) * half_dim;

    double local_min_ = std::numeric_limits<double>::max();
    double local_max_ = std::numeric_limits<double>::min();
    #pragma omp parallel for reduction(min:local_min_) reduction(max:local_max_)
    for (size_t y = rank_row_idx * half_dim; y < end_row_idx; y++)
    {
        for (size_t x = rank_col_idx * half_dim; x < end_col_idx; x++)
        {
            double cell_water = domain[y][x];
            double update = 0.0;

            // Add left neighbor
            if (x != 0)
            {
                double difference = domain[y][x - 1] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // Add right neighbor
            if (x != pd.dimension - 1)
            {
                double difference = domain[y][x + 1] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // Add lower neighbor
            if (y != 0)
            {
                double difference = domain[y - 1][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // Add upper neighbor
            if (y != pd.dimension - 1)
            {
                double difference = domain[y + 1][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            double waterheight = domain[y][x] + update;

            if (pd.use_second_domain)
            {
                pd.domain1[y][x] = waterheight;
            }
            else
            {
                pd.domain2[y][x] = waterheight;
            }


            if (waterheight > local_max_)
            {
                local_max_ = waterheight;
            }
            else if (waterheight < local_min_)
            {
                local_min_ = waterheight;
            }
        }
    }
    local_max = local_max_;
    local_min = local_min_;
}

bool termination_criteria_fulfilled(ProblemData &pd)
{
    double global_max, global_min;

    // TODO @Students:
    // track min/max across ranks
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    double diff = global_max - global_min;
    if ((global_max - global_min) < Utility::threshold)
    {
        return true;
    }

    return false;
}


double* create_buffer_from_col(double **arr, int row_offset, int col_idx, int size) {
    double *buffer = new double[size];
    for (int i = 0; i < size; i++) {
        buffer[i] = arr[row_offset + i][col_idx];
    }
    return buffer;
}

void exchange_halo(ProblemData &pd)
{
    // TODO @Students:
    // Implement halo exchange
    double **domain = Utility::get_domain(pd);
    int half_dim = pd.dimension / 2;
    int rank_row_idx = rank / NUM_PROCESS_PER_ROW;
    int rank_col_idx = rank % NUM_PROCESS_PER_ROW;

    // exchange rows
    if (rank/2 == 0) {
        MPI_Sendrecv(&(domain[half_dim - 1][rank_col_idx * half_dim]), half_dim, MPI_DOUBLE, rank + 2, 0, &(domain[half_dim][rank_col_idx * half_dim]), half_dim, MPI_DOUBLE, rank + 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Sendrecv(&(domain[half_dim][rank_col_idx * half_dim]), half_dim, MPI_DOUBLE, rank - 2, 0, &(domain[half_dim - 1][rank_col_idx * half_dim]), half_dim, MPI_DOUBLE, rank - 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // exchange columns
    double* recv_buffer = new double[half_dim];
    int row_offset = rank_row_idx * half_dim;
    if (rank % 2 == 0) {
        double* send_buffer = create_buffer_from_col(domain, row_offset, half_dim - 1, half_dim);
        MPI_Sendrecv(send_buffer, half_dim, MPI_DOUBLE, rank + 1, 0, recv_buffer, half_dim, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < half_dim; i++) {
            domain[row_offset + i][half_dim] = recv_buffer[i];
        }
        delete[] send_buffer;
    }
    else {
        double* send_buffer = create_buffer_from_col(domain, row_offset, half_dim, half_dim);
        MPI_Sendrecv(send_buffer, half_dim, MPI_DOUBLE, rank - 1, 0, recv_buffer, half_dim, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < half_dim; i++) {
            domain[row_offset + i][half_dim - 1] = recv_buffer[i];
        }
        delete[] send_buffer;
    }
    delete[] recv_buffer;

}

unsigned long long simulate(ProblemData &pd)
{
    volatile bool terminate_criteria_met = false;
    while (!terminate_criteria_met)
    {
        exchange_halo(pd);
        compute_stencil(pd);
        terminate_criteria_met = termination_criteria_fulfilled(pd);
        Utility::switch_arrays(pd);
        pd.patch_updates += 1;
    }
    return pd.patch_updates;
}


double *generate_initial_water_height_contiguous(ProblemData &pd, int seed)
{
    Utility::generator.seed(seed);
    size_t half_dimension = pd.dimension / 2;
    size_t y_offsets[2] = {0, half_dimension};
    size_t x_offsets[2] = {0, half_dimension};
    double *data = new double[pd.dimension * pd.dimension];
    int counter = 0;
    for (size_t yoff : y_offsets)
    {
        for (size_t xoff : x_offsets)
        {
            for (size_t y = 0 + yoff; y < half_dimension + yoff; y++)
            {
                for (size_t x = 0 + xoff; x < half_dimension + xoff; x++)
                {
                    data[counter] = Utility::get_water_height(y, x);
                    counter++;
                }
            }
        }
    }
    return data;
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    ProblemData pd;

    // TODO @Students:
    // Initialize MPI
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int seed = 0;
    
    // TODO @Students:
    // Think about minimizing the array size at each rank. 
    // this might require additional changes elsewhere
    Utility::init_problem_data(pd, Utility::domain_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *initial_water_heights;
    if (rank == 0) {
        Utility::readInput(seed);
        initial_water_heights = generate_initial_water_height_contiguous(pd, seed);
    }

    // TODO @Students:
    // Initialize MPI find a way to send the initial data to the domains of other MPI Ranks
    double *this_rank_water_heights = new double[pd.dimension * pd.dimension];
    size_t half_dimension = pd.dimension / 2;
    size_t sub_matrix_size = half_dimension * half_dimension;
    double *recvbuf = new double[sub_matrix_size];
    int displacement = rank * sub_matrix_size;
    MPI_Scatter(initial_water_heights, sub_matrix_size, MPI_DOUBLE, recvbuf, sub_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int rank_block_row_idx = rank / NUM_PROCESS_PER_ROW;
    int rank_block_col_idx = rank % NUM_PROCESS_PER_ROW;
    int row_idx_end = (rank_block_row_idx + 1) * half_dimension;
    int col_idx_end = (rank_block_col_idx + 1) * half_dimension;
    int counter = 0;
    for (int y = rank_block_row_idx * half_dimension; y < row_idx_end; y++) {
        for (int x = rank_block_col_idx * half_dimension; x < col_idx_end; x++) {
            this_rank_water_heights[y * pd.dimension + x] = recvbuf[counter];
            counter++;
        }
    }

    Utility::apply_initial_water_height(pd, this_rank_water_heights);

    simulate(pd);
    Utility::free_problem_data(pd);

    // TODO @Students:
    // Finalize MPI
    MPI_Finalize();
    if (rank == 0) {
        delete[] initial_water_heights;
        std::cout << pd.patch_updates << std::endl;
        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    }
    return 0;
}
