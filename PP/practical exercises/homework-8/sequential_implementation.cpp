#include <cstddef> // for size_t
#include "Utility.h"
#include <algorithm>

#include "mpi.h"
#include <chrono>


double local_min{0.0};
double local_max{0.0};

void compute_stencil(ProblemData &pd)
{
    // For evert cell with coordinates (y,x) compute the influx from neighbor cells
    // Apply reflecting boundary conditions
    local_min = std::numeric_limits<double>::max();
    local_max = std::numeric_limits<double>::min();
    double **domain = Utility::get_domain(pd);

    for (size_t y = 0; y < pd.dimension; y++)
    {
        for (size_t x = 0; x < pd.dimension; x++)
        {

            double cell_water = domain[y][x];
            double update = 0.0;

            if (x != 0)
            {
                double difference = domain[y][x - 1] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            if (x != pd.dimension - 1)
            {
                double difference = domain[y][x + 1] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            if (y != 0)
            {
                double difference = domain[y - 1][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            if (y != pd.dimension - 1)
            {
                double difference = domain[y + 1][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            double waterheight = cell_water + update;

            if (pd.use_second_domain)
            {
                pd.domain1[y][x] = waterheight;
            }
            else
            {
                pd.domain2[y][x] = waterheight;
            }

            if (waterheight > local_max)
            {
                local_max = waterheight;
            }
            else if (waterheight < local_min)
            {
                local_min = waterheight;
            }
        }
    }
}

bool termination_criteria_fulfilled(ProblemData &pd)
{
    double max = local_max;
    double min = local_min;

    double diff = max - min;
    std::cout << "Max: " << local_max << " Min: " << local_min << " Diff: " << diff << std::endl;
    if ((max - min) < Utility::threshold)
    {
        return true;
    }

    return false;
}

unsigned long long simulate(ProblemData &pd)
{
    volatile bool terminate_criteria_met = false;

    while (!terminate_criteria_met)
    {
        compute_stencil(pd);
        terminate_criteria_met = termination_criteria_fulfilled(pd);
        Utility::switch_arrays(pd);
        pd.patch_updates += 1;
    }

    return pd.patch_updates;
}

double *generate_initial_water_height(ProblemData &pd, int seed)
{
    Utility::generator.seed(seed);
    size_t half_dimension = pd.dimension / 2;
    size_t y_offsets[2] = {0, half_dimension};
    size_t x_offsets[2] = {0, half_dimension};

    double *data = new double[pd.dimension * pd.dimension];

    for (size_t yoff : y_offsets)
    {
        for (size_t xoff : x_offsets)
        {
            for (size_t y = 0 + yoff; y < half_dimension + yoff; y++)
            {
                for (size_t x = 0 + xoff; x < half_dimension + xoff; x++)
                {
                    data[y * pd.dimension + x] = Utility::get_water_height(y, x);
                }
            }
        }
    }

    return data;
}

int main()
{

    MPI_Init(0, 0);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Rank: " << rank << " Size: " << size << std::endl;

    if(rank == 0){
        auto start = std::chrono::high_resolution_clock::now();
        ProblemData pd;
        int seed = 0;
        // Utility::readInput(seed);
        Utility::init_problem_data(pd, Utility::domain_size);
        double *initial_water_heights = generate_initial_water_height(pd, seed);
        Utility::apply_initial_water_height(pd, initial_water_heights);
        delete[] initial_water_heights;

        simulate(pd);
        std::cout << pd.patch_updates << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";

        Utility::free_problem_data(pd);

    }
    MPI_Finalize();

    return 0;
}