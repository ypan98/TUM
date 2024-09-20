//
// Created by Vincent Bode on 21/04/2020.
// Updated by Dennis-Florian Herr 28/04/2022.
//

#ifndef ASSIGNMENTS_VV_AES_H
#define ASSIGNMENTS_VV_AES_H

#include <limits>

#define BLOCK_SIZE 8
#define UNIQUE_CHARACTERS 256
#define ROUNDS 10
#define ITERATIONS 400000

/*
 * This is the message given to you to encrypt to verify the correctness and speed of your approach. Unfortunately,
 * there are no state secrets to be found here. Those would have probably made a pretty penny on the black market.
 */
uint8_t message[BLOCK_SIZE][BLOCK_SIZE] = {
        {'T', 'H', 'I', 'S', ' ', 'I', 'S', ' '},
        {'A', ' ', 'V', 'E', 'R', 'Y', ' ', 'S'},
        {'E', 'C', 'R', 'E', 'T', ' ', 'M', 'E'},
        {'S', 'S', 'A', 'G', 'E', '!', ' ', 'T'},
        {'R', 'Y', ' ', 'T', 'O', ' ', 'C', 'R'},
        {'A', 'C', 'K', ' ', 'I', 'T', ' ', 'A'},
        {'S', ' ', 'Q', 'U', 'I', 'C', 'K', ' '},
        {'Y', 'O', 'U', ' ', 'C', 'A', 'N', '!'}
};

/*
 * The set of all keys generated at runtime and the index of the current key.
 */
int currentKey = 0;
uint8_t allKeys[ROUNDS][BLOCK_SIZE][BLOCK_SIZE];

/*
 * Use this like a 2D-Matrix key[BLOCK_SIZE][BLOCK_SIZE];
 * The key is only handed to you when it's time to actually encrypt something.
 */
uint8_t (*key)[BLOCK_SIZE];

/*
 * This is the source list of characters. If you wish to translate a character, find its index in this list. The
 * corresponding output character resides at the same index in the substituted character list.
 */
uint8_t originalCharacter[] = { 87, 141, 112, 19, 125, 139, 18, 197, 29, 254, 219, 241, 53, 70, 45, 12, 185, 102, 55, 8, 144, 23, 160, 188, 3, 43, 13, 127, 57, 251, 162, 135, 203, 206, 14, 234, 184, 137, 131, 31, 64, 96, 195, 38, 66, 166, 238, 92, 81, 0, 26, 6, 149, 211, 175, 1, 84, 33, 94, 233, 133, 118, 199, 104, 122, 85, 243, 154, 246, 93, 59, 240, 178, 78, 142, 187, 7, 79, 252, 194, 186, 207, 148, 82, 74, 50, 115, 176, 204, 72, 39, 10, 34, 155, 138, 182, 209, 134, 108, 165, 28, 232, 163, 47, 46, 69, 255, 114, 191, 140, 228, 15, 95, 83, 225, 63, 248, 76, 237, 183, 91, 103, 214, 65, 201, 239, 146, 247, 4, 22, 151, 223, 86, 230, 80, 126, 136, 44, 129, 56, 110, 168, 58, 210, 193, 51, 152, 60, 147, 21, 106, 119, 97, 37, 2, 25, 67, 170, 250, 9, 75, 98, 245, 48, 242, 222, 220, 73, 42, 157, 218, 107, 217, 20, 32, 212, 164, 113, 150, 68, 158, 192, 40, 71, 169, 159, 90, 109, 128, 208, 249, 24, 161, 105, 17, 226, 231, 189, 173, 156, 180, 88, 253, 143, 100, 99, 130, 124, 172, 132, 41, 16, 179, 190, 171, 145, 62, 61, 215, 35, 227, 244, 235, 11, 101, 236, 36, 49, 174, 216, 221, 120, 27, 177, 77, 213, 224, 200, 198, 117, 196, 54, 167, 121, 30, 205, 116, 111, 52, 229, 89, 153, 5, 123, 181, 202 };

/*
 * This is the output list of characters. If you wish to translate a character, find its index in the original list.
 * The corresponding output character resides at the same index in this list.
 */
uint8_t substitutedCharacter[] = { 113, 128, 187, 74, 29, 59, 9, 122, 150, 168, 49, 12, 62, 24, 30, 248, 35, 40, 130, 107, 158, 180, 172, 212, 237, 249, 215, 175, 223, 78, 143, 200, 114, 126, 219, 164, 138, 243, 52, 244, 1, 109, 192, 66, 71, 157, 169, 195, 214, 45, 31, 252, 245, 63, 22, 239, 16, 6, 199, 235, 112, 207, 118, 250, 98, 139, 19, 197, 69, 119, 2, 87, 56, 18, 184, 182, 227, 92, 89, 222, 203, 84, 61, 104, 38, 133, 57, 179, 134, 27, 142, 101, 43, 174, 102, 11, 228, 103, 206, 8, 21, 42, 148, 68, 10, 117, 205, 209, 176, 196, 53, 111, 173, 70, 80, 135, 77, 247, 7, 4, 188, 178, 94, 33, 13, 127, 91, 88, 255, 241, 159, 86, 96, 216, 51, 226, 221, 132, 125, 183, 185, 186, 236, 67, 73, 100, 170, 34, 58, 90, 151, 47, 85, 46, 105, 161, 165, 97, 136, 115, 108, 26, 137, 254, 190, 0, 39, 79, 82, 116, 166, 75, 162, 240, 41, 55, 25, 160, 141, 72, 225, 20, 171, 232, 120, 140, 198, 23, 204, 50, 251, 177, 163, 28, 191, 156, 242, 167, 220, 129, 210, 234, 147, 121, 253, 17, 181, 146, 110, 64, 238, 149, 145, 48, 60, 81, 218, 154, 36, 233, 229, 83, 155, 213, 76, 37, 224, 123, 246, 189, 99, 153, 54, 193, 95, 5, 230, 15, 65, 3, 152, 211, 231, 14, 202, 32, 144, 44, 217, 124, 106, 194, 93, 201, 131, 208 };

 uint8_t polynomialCoefficients[BLOCK_SIZE][BLOCK_SIZE] = {
        { 1, 2, 3, 4, 5, 6, 7, 8,},
        { 8, 7, 6, 5, 4, 3, 2, 1},
        { 1, 2, 3, 4, 5, 6, 7, 8},
        { 8, 7, 6, 5, 4, 3, 2, 1},
        { 8, 8, 8, 8, 8, 8, 8, 8},
        { 4, 5, 6, 8, 6, 4, 2, 4},
        { 8, 3, 4, 6, 5, 1, 1, 2}
};

/*
 * Generate some keys that can be used for encryption based on the seed set from the input.
 */
void generate_keys() {
    // Fill the key block
    for(auto & currentKey : allKeys) {
        for (auto & row : currentKey) {
            for (unsigned char & column : row) {
                column = rand() % std::numeric_limits<uint8_t>::max();
            }
        }
    }
}

void readInput() {
    std::cout << "READY" << std::endl;
    unsigned int seed = 0;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if(seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    // Set the pseudo random number generator seed used for generating encryption keys
    srand(seed);

    generate_keys();
}

void writeOutput() {
    // Output the current state of the message in hexadecimal.
    for (int row = 0; row < BLOCK_SIZE; row++) {
        std::cout << std::hex << (int) message[row][0] << (int) message[row][1] << (int) message[row][2]
                  << (int) message[row][3];
    }
    // This stops the timer.
    std::cout << std::endl << "DONE" << std::endl;
}

/*
 * This is a utility method. It determines the next key to use from the set of pre-generated keys. In a real
 * cryptographic system, the subsequent keys are generated from a key schedule from the original key. To keep the code
 * short, we have omitted this behavior.
 */
void set_next_key() {
    key = &allKeys[currentKey][0];
    currentKey = (currentKey + 1) % ROUNDS;
}

#endif //ASSIGNMENTS_VV_AES_H
