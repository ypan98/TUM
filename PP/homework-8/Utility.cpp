#include "Utility.h"

namespace Utility {
 std::mt19937 generator = std::mt19937();
}

void Utility::readInput(int &seed)
{
    seed = 0;
    std::cout << "READY" << std::endl;
    std::cin >> seed;
}

double Utility::get_water_height(size_t y, size_t x)
{

    unsigned int xdisloc = xmid - x;
    unsigned int ydisloc = ymid - y;

    if (-offset_radius < xdisloc && offset_radius > xdisloc && -offset_radius < ydisloc && offset_radius > ydisloc)
    {
        return higher(generator);
    }

    return lower(generator);
}

void Utility::free_problem_data(ProblemData &pd)
{
    for (size_t i = 0; i < pd.dimension; i++)
    { // allocate the second arrays
        delete[] pd.domain1[i];
        delete[] pd.domain2[i];
    }

    delete[] pd.domain1;
    delete[] pd.domain2;
}

void Utility::init_problem_data(ProblemData &pd, size_t domain_size)
{
    pd.dimension = domain_size;

    pd.domain1 = new double *[pd.dimension]; // allocate the first array
    pd.domain2 = new double *[pd.dimension];

    for (size_t i = 0; i < pd.dimension; i++)
    { // allocate the second arrays
        pd.domain1[i] = new double[pd.dimension];
        pd.domain2[i] = new double[pd.dimension];
    }
}

void Utility::apply_initial_water_height(ProblemData &pd, double *initial_data)
{
    for (size_t y = 0; y < pd.dimension; y++)
    {
        for (size_t x = 0; x < pd.dimension; x++)
        {
            pd.domain1[y][x] = initial_data[y * pd.dimension + x];
        }
    }
}
