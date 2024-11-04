#include "Utility.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1

/* 
 * Dense matrix vector multiplication
 */
void dmv(float* mat, float* in_vec, float *out_vec, size_t mat_size){
    float sum = 0;
    for(size_t row = 0; row < mat_size; row++){
        out_vec[row] = 0;
        for(size_t col = 0; col < mat_size; col++){
            out_vec[row] += mat[col + row * mat_size] * in_vec[col];
        }
        sum += out_vec[row];
    }

    // normalize output vector
    for(size_t row = 0; row < mat_size; row++){
        out_vec[row] /= sum;
    }
}

int main()
{
    unsigned int seed = readInput();
    float *mat = new float[MAT_SIZE * MAT_SIZE];
    float *in_vec = new float[MAT_SIZE];
    float *out_vec = new float[MAT_SIZE];

#ifdef PRINT_TIME
    TicToc total_time;
#endif
    generate_test(seed, mat, in_vec);

    for(int i = 0; i < ITER_NUM; i++){
        dmv(mat, in_vec, out_vec, MAT_SIZE);
        out_vec[i] = 1.0/double(i + 1);
        memcpy(in_vec, out_vec, sizeof(float) * MAT_SIZE);
    }

    outputResult(seed, out_vec);
#ifdef PRINT_TIME
    std::cerr << "time used: " << total_time.toc() << "ms.\n";
#endif
}
