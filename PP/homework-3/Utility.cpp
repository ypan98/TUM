//
// Created by Vincent_Bode on 10.06.2020.
//

#include <iostream>
#include <random>
#include <cstring>
#include "Utility.h"
#include <getopt.h>

#include <openssl/sha.h>

// generate a sha1 hash from a std::string
Sha1Hash Utility::sha1(std::string& input){
   return sha1(reinterpret_cast<const unsigned char*>(input.data()), input.length());
}

// generate a new Sha1Hash from a Sha1Hash
Sha1Hash Utility::sha1(Sha1Hash& input){
   return sha1(input.data,  SHA1_SIZE);
}

// generate a new Sha1 hash by concatenating 2 hashes
Sha1Hash Utility::sha1(Sha1Hash& hash1, Sha1Hash& hash2){

   unsigned char* combinedInput = static_cast<unsigned char*>(malloc( SHA1_SIZE * 2));
   memcpy(combinedInput, hash1.data,  SHA1_SIZE);
   memcpy(&combinedInput[20], hash2.data,  SHA1_SIZE);
   Sha1Hash result = sha1(combinedInput,  SHA1_SIZE*2);
   free(combinedInput);
   return result;
}

// Wrapper method for creating a Sha1Hash struct using the openssl SHA1 method.
Sha1Hash Utility::sha1(const unsigned char* input, size_t input_length){
    Sha1Hash output;

    // call openssl sha1 method. this requires compilation with -lcrypto
    SHA1(input, input_length, output.data);

    return output;
}

int Utility::count_leading_zero_bits(Sha1Hash& hash){

    unsigned int bits;
    int total_leading_zero_bits = 0;

    for(int i = 0; i <  SHA1_SIZE; i+=4){
        // adjust bit order due to little endianesss
        bits = (static_cast<unsigned int>(hash.data[i]) << 24) |
            (static_cast<unsigned int>(hash.data[i+1]) << 16) |
            (static_cast<unsigned int>(hash.data[i+2]) << 8) |
            (static_cast<unsigned int>(hash.data[i+3]));
        if(bits == 0){
            total_leading_zero_bits += 32;
        }else{
            // __builtin_clz is undefined for 0
            return total_leading_zero_bits + __builtin_clz(bits);
        }
    }

    return total_leading_zero_bits;
}

unsigned int Utility::readInput() {
    unsigned int seed = 0;
    std::cout << "READY" << std::endl;
    std::cin >> seed;

    return seed;
}

void Utility::printHash(Sha1Hash& hash, bool newLine) {

    for (int i = 0; i <  SHA1_SIZE; i++) {
        printf("%02x", hash.data[i]);
    }
    if(newLine) printf("\n");
}

void Utility::parse_input(int& numProblems, int& leadingZerosProblem, int& leadingZerosSolution, int argc, char** argv) {
    int option;
    while ((option = getopt(argc, argv, "hn:p:s:")) != -1) {
        switch (option) {
        case 'p':
            leadingZerosProblem = strtol(optarg, nullptr, 0);
            if (leadingZerosProblem <= 0) {
                std::cerr << "Error parsing leading zeros solution." << std::endl;
                exit(-1);
            }
            break;
        case 's':
            leadingZerosSolution = strtol(optarg, nullptr, 0);
            if (leadingZerosSolution <= 0) {
                std::cerr << "Error parsing leading zeros problem." << std::endl;
                exit(-1);
            }
            break;
        case 'n':
            numProblems = strtol(optarg, nullptr, 0);
            if (numProblems <= 0) {
                std::cerr << "Error parsing number problems." << std::endl;
                exit(-1);
            }
            break;
        case 'h':
            std::cerr << "Usage: " << argv[0] << " [-v] [-p] [-n <numProblems>] [-h]" << std::endl
                << "-s: Count of leading zero bits in the solution hashes"
                << std::endl
                << "-p: Count of leading zero bits in the problem hashes"
                " and uses lots of memory." << std::endl
                << "-n: The number of problems to solve." << std::endl
                << "-h: Show this help topic." << std::endl;
            exit(-1);
        default:
            std::cerr << "Unknown option: " << (unsigned char)option << std::endl;
            exit(-1);
        }
    }

    std::cerr << "Solving " << numProblems <<
        " problems (leading zero bits problem: " << leadingZerosProblem << ", leading zero bits soltuion "
        << leadingZerosSolution << ")" << std::endl;

}