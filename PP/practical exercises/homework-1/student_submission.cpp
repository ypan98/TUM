#include <cstring>
#include <iostream>

#include "vv-aes.h"

/**
 * This function takes the characters stored in the 7x7 message array and substitutes each character for the
 * corresponding replacement as specified in the originalCharacter and substitutedCharacter array.
 * This corresponds to step 2.1 in the VV-AES explanation.
 * 
 * Modification: aux dict to speedup index finding process and cache friendly traversal
 */
void substitute_bytes(uint8_t* dict) {
    // For each byte in the message
    for (int row = 0; row < BLOCK_SIZE; row++) {
        for (int column = 0; column < BLOCK_SIZE; column++) {
            message[row][column] = dict[message[row][column]];
        }
    }
}

/*
 * This function shifts each row by the number of places it is meant to be shifted according to the AES specification.
 * Row zero is shifted by zero places. Row one by one, etc.
 * This corresponds to step 2.2 in the VV-AES explanation.
 * 
 * Modification: Avoid allocating/freeing memory several times for each row
 */
void shift_rows(uint8_t aux_container[BLOCK_SIZE][BLOCK_SIZE], uint8_t shifted_indices[BLOCK_SIZE][BLOCK_SIZE]) {
    // Shift each row, where the row index corresponds to how many columns the data is shifted.
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            aux_container[i][j] = message[i][shifted_indices[i][j]];
        }
    }

    memcpy(message, aux_container, BLOCK_SIZE * BLOCK_SIZE);
}

/*
 * This function calculates x^n for polynomial evaluation.
 *
 * Modification: Fast Exponentiation
 */
uint8_t power(uint8_t x, int n) {
    // Calculates x^n
    if (n == 1) {
        return x;
    }
    else if (n % 2 == 0) {
        return power(x, n/2) * power(x, n/2);
    }
    return x * power(x, (n - 1)/2) * power(x, (n - 1)/2);
}

/*
 * For each column, mix the values by evaluating them as parameters of multiple polynomials.
 * This corresponds to step 2.3 in the VV-AES explanation.
 * 
 * Modification: pre-compute powers
 */
void mix_columns(uint8_t power_dict[][UNIQUE_CHARACTERS]) {
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0; column < BLOCK_SIZE; ++column) {
            uint8_t result = 0;
            for (int degree = 0; degree < BLOCK_SIZE; degree++) {
                result += polynomialCoefficients[row][degree] * power_dict[degree][message[degree][column]];
            }
            message[row][column] = result;
        }
    }
}

/*
 * Add the current key to the message using the XOR operation.
 * 
 * Modification: Cache-friendly matrix traversal
 */
void add_key() {
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0; column < BLOCK_SIZE; ++column) {
            // ^ == XOR
            message[row][column] = message[row][column] ^ key[row][column];
        }
    }
}

/*
 * Your main encryption routine.
 */
int main() {
    // Receive the problem from the system.
    readInput();

    // create a dict to speedup index finding process
    uint8_t dict[UNIQUE_CHARACTERS];

    for (int i = 0; i < UNIQUE_CHARACTERS; i++) {
        dict[originalCharacter[i]] = substitutedCharacter[i];
    }

    // create a dict to speedup power computation
    uint8_t power_dict[BLOCK_SIZE][UNIQUE_CHARACTERS];

    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < UNIQUE_CHARACTERS; j++) {
            power_dict[i][j] = power(j, i + 1);
        }
    }
    
    // create a container to do shift
    uint8_t aux_container[BLOCK_SIZE][BLOCK_SIZE];
    uint8_t shifted_indices[BLOCK_SIZE][BLOCK_SIZE];

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        int idx = 0;

        for (int j = i; j < BLOCK_SIZE; ++j) {
            shifted_indices[i][idx++] = j;
        }
        for (int j = 0; j < i; ++j) {
            shifted_indices[i][idx++] = j;
        }
    }    
    // For extra security (and because Varys wasn't able to find enough test messages to keep you occupied) each message
    // is put through VV-AES lots of times. If we can't stop the adverse Maesters from decrypting our highly secure
    // encryption scheme, we can at least slow them down.
    for (int i = 0; i < ITERATIONS; i++) {
        // For each message, we use a predetermined key (e.g. the password). In our case, its just pseudo random.
        set_next_key();

        // First, we add the key to the message once:
        add_key();

        // We do 9+1 rounds for 128 bit keys
        for (int round = 0; round < ROUNDS; round++) {
            //In each round, we use a different key derived from the original (refer to the key schedule).
            set_next_key();

            // These are the four steps described in the slides.
            substitute_bytes(dict);
            shift_rows(aux_container, shifted_indices);
            mix_columns(power_dict);
            add_key();
        }
        // Final round
        substitute_bytes(dict);
        shift_rows(aux_container, shifted_indices);
        add_key();
    }

    // Submit our solution back to the system.
    writeOutput();

    return 0;
}
