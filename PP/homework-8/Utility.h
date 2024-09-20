#pragma once

#include <cstddef> // for size_t
#include <random>  // for the random engine and distributions
#include <string>
#include <iostream>

struct ProblemData
{
    // Allocated data needed to solve the problem
    size_t dimension{0};
    double **domain1{nullptr}; // ~ dimension x dimension x 4 bytes
    double **domain2{nullptr}; // ~ dimension x dimension x 4 bytes

    bool use_second_domain{false};

    unsigned long long patch_updates{0};

};

namespace Utility
{
    // Variables required for the problem definition and generation
    constexpr double threshold = 0.1;
    constexpr double viscosity_factor = 4.0;
    constexpr size_t domain_size = 1000;
    constexpr size_t print_frequency = 200;

    constexpr unsigned int xmid = domain_size / 2;
    constexpr unsigned int ymid = domain_size / 2;
    constexpr unsigned int offset_radius = domain_size / 8;

    // Random number generator and distribution
    extern std::mt19937 generator;
    static std::uniform_real_distribution<double> higher(35, 45);
    static std::uniform_real_distribution<double> lower(15, 25);

    void readInput(int &seed);
    double get_water_height(size_t y, size_t x);
    void init_problem_data(ProblemData &pd, size_t domain_size);
    void free_problem_data(ProblemData &pd);
    void apply_initial_water_height(ProblemData &pd, double *initial_data);

    inline void switch_arrays(ProblemData &pd)
    {
        pd.use_second_domain = !pd.use_second_domain;
    }

    inline double **get_domain(ProblemData &pd)
    {
        double **d = pd.use_second_domain ? pd.domain2 : pd.domain1;
        return d;
    }

    inline double **get_new_domain(ProblemData &pd)
    {
        double **d = pd.use_second_domain ? pd.domain1 : pd.domain2;
        return d;
    }
};