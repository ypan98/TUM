#ifndef DGEMM_H
#define DGEMM_H

#include <cstring>
#include <iostream>
#include <random>

#define MATRIX_SIZE 2023
#define NUM_ELEMENTS MATRIX_SIZE * MATRIX_SIZE
#define MEM_SIZE NUM_ELEMENTS * sizeof(float)

void generateProblemFromInput(float& alpha, float* a, float* b, float& beta, float* c) {
    unsigned int seed = 0;
    std::cout << "READY" << std::endl;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if (seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-5, 5);

    /* initialisation */
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            *(a + i * MATRIX_SIZE + j) = distribution(random);
            *(b + i * MATRIX_SIZE + j) = distribution(random);
        }
    }

    for(int i = 0; i < NUM_ELEMENTS; i++){
        *(c + i) = distribution(random);
    }

    alpha = distribution(random);
    beta = distribution(random);
}


void outputSolution(const float* c) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < NUM_ELEMENTS; ++i) {
        sum += c[i] * ((i+1) % 283);

    }
    std::cout << "Sum of final matrix values: " << sum << std::endl;
    std::cout << "DONE" << std::endl;
}

#endif // DGEMM_H
