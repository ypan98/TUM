//
// Created by Vincent_Bode on 10.06.2020.
//

#ifndef ASSIGNMENTS_UTILITY_H
#define ASSIGNMENTS_UTILITY_H

#include <cstring>
#define  SHA1_SIZE 20

struct Sha1Hash {
    unsigned char data[ SHA1_SIZE];
};

class Utility {
public:
    static void parse_input(int& numProblems, int& leadingZerosProblem, int& leadingZerosSolution, int argc, char** argv);
    static Sha1Hash sha1(std::string& input);
    static Sha1Hash sha1(Sha1Hash& input);
    static Sha1Hash sha1(Sha1Hash& hash1, Sha1Hash& hash2);
    static Sha1Hash sha1(const unsigned char* input, size_t input_length);
    static int count_leading_zero_bits(Sha1Hash& hash);
    static void printHash(Sha1Hash& hash, bool newLine = true);
    static unsigned int readInput();
};

#endif //ASSIGNMENTS_UTILITY_H
