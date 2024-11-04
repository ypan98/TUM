#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>

#define NUM_DATAPOINTS 50000
#define NUM_PREDICTIONS 10000
#define NUM_CLASSES 10
#define K 100
#define ACTIVE_SQUARE_LENGTH 100

double datapoints[NUM_DATAPOINTS][2];
int datapoint_classes[NUM_DATAPOINTS];

/*
 * This function outputs the execution time. 
 */
class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

/*
 * Initializes seed for randomized testing.
 */
static unsigned int readInput()
{
    std::cout << "READY" << std::endl;
    unsigned int seed = 0;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if (seed == 0)
    {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    return seed;
}

/*
 * Generates random tests.
 */
static void generate_test(unsigned int seed, double input_data[][2])
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.

    for (int i = 0; i < NUM_DATAPOINTS; i++)
    {
        datapoints[i][0] = ACTIVE_SQUARE_LENGTH * double(generator()) / double(generator.max());
        datapoints[i][1] = ACTIVE_SQUARE_LENGTH * double(generator()) / double(generator.max());
        datapoint_classes[i] = generator() % NUM_CLASSES;
    }

    for (int i = 0; i < NUM_PREDICTIONS; i++)
    {
        input_data[i][0] = ACTIVE_SQUARE_LENGTH * double(generator()) / double(generator.max());
        input_data[i][1] = ACTIVE_SQUARE_LENGTH * double(generator()) / double(generator.max());
    }
}

static void init_class_num(int *each_class_num){
    for (int i = 0; i < NUM_CLASSES; i++){
        each_class_num[i] = 0;
    }
}

/*
 * Calculates euclidean distance between two points.
 */
double get_euclidean_distance(double x1, double y1, double x2, double y2)
{
    return sqrt((y1 - y2)*(y1 - y2) + (x1 - x2) * (x1 - x2));
}

/*
 * Finds k nearest neighbors by iterating through all neighbors and finding k ones with smallest euclidean distance.
 * This works the following way:
 * 1. We set the first K datapoints as our k nearest neighbors and we calculate the furthest one to our input points among these neighbors.
 * 2. We iterate through all other datapoints and check if their distance is smaller that the one from the furthest point.
 * 3. If we find such a point, we throw out the furthest point from our neighbors and make this new points a neighbor.
 * 4. We recompute the furthest point among our neighbors to have a next candidate for elimination.
 * 5. This process is repeated until all datapoints have been iterated through.
 */
void get_k_nearest_neighbors(double *point, int *k_nearest_neighbors)
{
    double min_distances[K];
    int max_index = 0;
    double max_distance = get_euclidean_distance(point[0], point[1], datapoints[0][0], datapoints[0][1]);

    //Set first K points as nearest neighbors
    for(int i = 0; i < K; i++)
    {
        min_distances[i] = get_euclidean_distance(point[0], point[1], datapoints[i][0], datapoints[i][1]);
        k_nearest_neighbors[i] = datapoint_classes[i];
        
        //find the biggest value among the neighbors (i.e. the candidate for elimination)
        if (max_distance < min_distances[i])
        {
            max_distance = min_distances[i];
            max_index = i;
        }
    }

    double distance;

    for(int i = K; i < NUM_DATAPOINTS; i++)
    {
        distance = get_euclidean_distance(point[0], point[1], datapoints[i][0], datapoints[i][1]);
        
        //Eliminate max_distance point if there is a closer one
        if (distance < max_distance)
        {
            min_distances[max_index] = distance;
            k_nearest_neighbors[max_index] = datapoint_classes[i];

            //Calculate new max_distance point
            max_index = 0;
            max_distance = min_distances[0];
            for(int j = 1; j < K; j++)
            {
                if (max_distance < min_distances[j])
                {
                    max_distance = min_distances[j];
                    max_index = j;
                }
            }

        }
    }
}

/*
 * Calculates the k-nn class of a point from the classes to which its neighbors were assigned,
 * by finding out which class is the majority one among the neighbors.
 */
void get_class_from_neighbors(int *k_nearest_neighbors, int *output)
{
    int class_points[NUM_CLASSES];

    //Initialize class points
    for(int i = 0; i < NUM_CLASSES; i++)
    {
        class_points[i] = 0;
    }

    //Increment class_points at the index corresponding to the class of a specific neighbor
    for(int i = 0; i < K; i++)
    {
        class_points[k_nearest_neighbors[i]]++;
    }

    *output = 0;
    int max_count = class_points[0];

    //Iterate through the class_points, find the majority one and set the output to this class
    for(int i = 1; i < NUM_CLASSES; i++)
    {
        if (class_points[i] > max_count)
        {
            max_count = class_points[i];
            *output = i;
        }
    }
}

/*
 * This function outputs the result. 
 */
static void outputResult(int *each_class_num)
{
    for (int i = 0; i < NUM_CLASSES; i++)
        std::cout << "Class " << i << " number: " << each_class_num[i] << std::endl;
    std::cout << "DONE" << std::endl;
}

#endif // UTILITY_H
