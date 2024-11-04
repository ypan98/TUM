#include "Utility.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

// uncomment this line to print used time
// comment this line before submission
#define PRINT_TIME 1

/*
 * Predicts the classes of NUM_PREDICTIONS points using k-nn.
 */
void predict(double input_data[][2], int *predicted_class, int *each_class_num)
{
    int k_nearest_neighbors[NUM_PREDICTIONS][K];

    for(int i = 0; i < NUM_PREDICTIONS; i++)
    {
        get_k_nearest_neighbors(input_data[i], k_nearest_neighbors[i]);
        get_class_from_neighbors(k_nearest_neighbors[i], &predicted_class[i]);
        each_class_num[predicted_class[i]]++;
    }
}

int main()
{
    double input_data[NUM_PREDICTIONS][2];
    int predicted_class[NUM_PREDICTIONS];
    int each_class_num[NUM_CLASSES] = {0};
    unsigned int seed = readInput();

#ifdef PRINT_TIME
    TicToc total_time;
#endif

    generate_test(seed, input_data);
    init_class_num(each_class_num);
    predict(input_data, predicted_class, each_class_num);
    outputResult(each_class_num);
#ifdef PRINT_TIME
    std::cerr << "time used: " << total_time.toc() << "ms.\n";
#endif
}