#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <stdlib.h>  
#include <random>
#include <chrono>

#define NUM_FACTORS 8192

using Integer = std::vector<int>;

Integer createInteger(int n)
{
    Integer res;
    res.clear();
    while (n)
    {
        res.push_back(n % 10);
        n /= 10;
    }
    return res;
}

Integer mulInteger(const Integer &a, const Integer &b);
/*
 * This is the main function that does work.
 * Here the numbers from the problem array are initialized with the datatype Integer.
 * Then the numbers are multiplied like in a tree, from the leaves to the root.
 * The leaves are the initial numbers, and the parent of 2 nodes is their product.
 * The tree is generated from the bottom to the top and the final result is located in the root.
 */
Integer calcProduct(int *problem)
{
    std::vector<Integer> int_list(NUM_FACTORS);

    //Initialize the numbers as Integer
    for (size_t i = 0; i < NUM_FACTORS; i++)
    {
        int_list[i] = createInteger(problem[i]);
    }

    //Go through all nodes in the tree level by level
    for (size_t inc = 1; inc < NUM_FACTORS; inc *= 2)
    {
        for (size_t i = 0; i < NUM_FACTORS; i += (2 * inc))
        {
            //Multiply 2 integers and store the result in place of the first factor
            int_list[i] = mulInteger(int_list[i], int_list[i + inc]);
        }
    }

    return int_list[0];
}

/*
 * This function adds two Integers digit by digit (like in school).
 */
Integer addInteger(const Integer &a, const Integer &b)
{
    Integer result;
    size_t ls = a.size();
    size_t rs = b.size();
    auto &longer = ls > rs ? a : b;
    auto &shorter = ls <= rs ? a : b;

    int carry = 0;
    for (size_t i = 0; i < longer.size(); i++)
    {
        int ad = i < shorter.size() ? shorter[i] : 0;
        int tmp = ad + longer[i] + carry;
        carry = tmp / 10;
        result.push_back(tmp % 10);
    }
    if (carry)
    {
        result.push_back(carry);
    }
    return result;
}

/* the mechanism of this function. Example:
 *      a = 1234
 *      b = 2
 *      shift = 2
 * 
 *        1 2 3 4
 *      X   2      <- shift left 2 digit
 *      ----------
 *    2 4 6 8 0 0
 */
Integer mulShiftedInteger(const Integer &a, int b, int shift)
{
    Integer res;
    for (int i = 0; i < shift; i++)
    {
        res.push_back(0);
    }
    int carry = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        int tmp = a[i] * b + carry;
        if (i + shift >= res.size())
        {
            res.push_back(tmp % 10);
        }
        else
        {
            tmp += res[i + shift];
        }
        carry = tmp / 10;
    }
    if (carry)
    {
        res.push_back(carry);
    }
    return res;
}

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
static void
generate_test(unsigned int seed, int *problem)
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.

    for(int i = 0; i < NUM_FACTORS; i++)
    {
        problem[i] = generator() % NUM_FACTORS + 1;
    }
}

/*
 * This function outputs the result. 
 */
static void outputResult(Integer &n)
{
    for (size_t i = 0; i < 100; i++)
    {
        std::cout << n[n.size() - i - 1];
    }
    std::cout << std::endl;

    std::cout << "DONE" << std::endl;
}

#endif // UTILITY_H
