#include "Utility.h"
#include <chrono>

/*
 * Our message is encrypted/decrypted NUM_ITERATIONS times (in main).
 * At each iteration one layer is decrypted using this function and the result is stored in the same decryptedMessage array.
 * In order to decrypt a message, the location of each character from decryptedMessage in the keys array is found.
 * The decrypted character is located at the same index in the values array (as in keys).
 */
void decrypt_message(uint8_t *decryptedMessage, uint8_t *keys, uint8_t *values)
{
    // TODO: modify the following code to improve the efficiency
    int index = -1;
    for (unsigned int i = 0; i < STRING_LEN; ++i)
    {
        for (int k = 0; k < UNIQUE_CHARACTERS; ++k)
        {
            if (decryptedMessage[i] == keys[k])
            {
                index = k;
            }
        }

        decryptedMessage[i] = values[index];
    }

    // End of TODO
}

// You don't need to modify the main function
int main()
{
    //This is a container for the decrypted message
    uint8_t decryptedMessage[STRING_LEN];

    /*
    * This is the source list of characters. If you wish to translate a character, find its index in this list. The
    * corresponding output character resides at the same index in the substituted character list.
    */
    uint8_t keys[256];

    /*
    * This is the output list of characters. If you wish to translate a character, find its index in the original list.
    * The corresponding output character resides at the same index in this list.
    */
    uint8_t values[256];


    unsigned int seed = readInput();

    /*
     * If you want to test the running time locally, uncomment the lines below.
     * IMPORTANT: THOSE LINES SHOULD BE COMMENTED OUT BEFORE SUBMISSION, i.e. the last output should be from output_message.
    */
    //std::chrono::high_resolution_clock::time_point start, stop;
    //start = std::chrono::high_resolution_clock::now();
    
    generate_test(decryptedMessage, keys, values, seed);

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        decrypt_message(decryptedMessage, keys, values);
    }
    
    output_message(decryptedMessage);
    
    //stop = std::chrono::high_resolution_clock::now();
    //int time_in_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    //std::cout << std::dec << "Operations executed in " << time_in_microseconds << " microseconds" << std::endl;
}