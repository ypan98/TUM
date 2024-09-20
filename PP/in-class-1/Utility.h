#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string.h>
#include <random>

#define STRING_LEN 21
#define NUM_ITERATIONS 50000
#define UNIQUE_CHARACTERS 256

/*
 * This function outputs the decryptedMessage. 
 */
static void output_message(uint8_t *decryptedMessage)
{
    std::cout << std::hex;
    for (int i = 0; i < STRING_LEN; i++)
    {
        std::cout << (int)decryptedMessage[i];
    }

    std::cout << std::endl
              << "DONE" << std::endl;
}

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
static void generate_test(uint8_t *decryptedMessage, uint8_t *originalCharacter, uint8_t *substitutedCharacter, unsigned int seed)
{
    std::minstd_rand0 generator(seed); // linear congruential random number generator.
    std::vector<std::pair<uint8_t, uint8_t>> characters = {std::make_pair(210, 109), std::make_pair(223, 244), std::make_pair(147, 230), std::make_pair(111, 56), std::make_pair(187, 18), std::make_pair(107, 232), std::make_pair(193, 199), std::make_pair(11, 81), std::make_pair(109, 116), std::make_pair(246, 213), std::make_pair(198, 11), std::make_pair(89, 208), std::make_pair(133, 188), std::make_pair(135, 79), std::make_pair(192, 209), std::make_pair(227, 91), std::make_pair(99, 19), std::make_pair(66, 71), std::make_pair(160, 99), std::make_pair(69, 131), std::make_pair(110, 36), std::make_pair(48, 9), std::make_pair(228, 185), std::make_pair(79, 186), std::make_pair(108, 80), std::make_pair(142, 133), std::make_pair(186, 159), std::make_pair(165, 245), std::make_pair(206, 15), std::make_pair(129, 173), std::make_pair(28, 118), std::make_pair(150, 26), std::make_pair(117, 190), std::make_pair(97, 59), std::make_pair(171, 222), std::make_pair(58, 197), std::make_pair(178, 95), std::make_pair(244, 76), std::make_pair(62, 182), std::make_pair(30, 216), std::make_pair(84, 87), std::make_pair(24, 106), std::make_pair(243, 10), std::make_pair(23, 157), std::make_pair(35, 24), std::make_pair(76, 64), std::make_pair(182, 107), std::make_pair(226, 13), std::make_pair(155, 211), std::make_pair(188, 205), std::make_pair(184, 35), std::make_pair(105, 78), std::make_pair(149, 142), std::make_pair(177, 21), std::make_pair(26, 105), std::make_pair(161, 39), std::make_pair(74, 69), std::make_pair(127, 179), std::make_pair(159, 113), std::make_pair(18, 33), std::make_pair(234, 161), std::make_pair(1, 67), std::make_pair(94, 252), std::make_pair(65, 212), std::make_pair(185, 137), std::make_pair(39, 145), std::make_pair(41, 128), std::make_pair(236, 100), std::make_pair(220, 125), std::make_pair(156, 127), std::make_pair(88, 165), std::make_pair(21, 147), std::make_pair(128, 242), std::make_pair(213, 183), std::make_pair(73, 122), std::make_pair(154, 207), std::make_pair(55, 121), std::make_pair(37, 219), std::make_pair(203, 202), std::make_pair(7, 154), std::make_pair(90, 61), std::make_pair(103, 254), std::make_pair(175, 28), std::make_pair(86, 250), std::make_pair(254, 236), std::make_pair(174, 174), std::make_pair(233, 101), std::make_pair(172, 246), std::make_pair(25, 255), std::make_pair(2, 251), std::make_pair(248, 54), std::make_pair(199, 126), std::make_pair(218, 206), std::make_pair(68, 112), std::make_pair(191, 52), std::make_pair(235, 139), std::make_pair(216, 86), std::make_pair(102, 16), std::make_pair(140, 1), std::make_pair(121, 248), std::make_pair(32, 152), std::make_pair(115, 41), std::make_pair(162, 88), std::make_pair(71, 243), std::make_pair(224, 130), std::make_pair(138, 192), std::make_pair(22, 46), std::make_pair(95, 184), std::make_pair(249, 0), std::make_pair(20, 96), std::make_pair(229, 3), std::make_pair(196, 97), std::make_pair(67, 34), std::make_pair(197, 234), std::make_pair(225, 239), std::make_pair(241, 6), std::make_pair(98, 90), std::make_pair(17, 98), std::make_pair(101, 134), std::make_pair(136, 44), std::make_pair(118, 215), std::make_pair(141, 114), std::make_pair(217, 135), std::make_pair(252, 231), std::make_pair(93, 191), std::make_pair(130, 167), std::make_pair(82, 2), std::make_pair(122, 50), std::make_pair(219, 148), std::make_pair(96, 8), std::make_pair(237, 187), std::make_pair(176, 12), std::make_pair(57, 94), std::make_pair(50, 72), std::make_pair(114, 23), std::make_pair(195, 166), std::make_pair(64, 193), std::make_pair(126, 240), std::make_pair(240, 25), std::make_pair(113, 85), std::make_pair(85, 196), std::make_pair(146, 214), std::make_pair(52, 84), std::make_pair(6, 226), std::make_pair(77, 117), std::make_pair(201, 29), std::make_pair(80, 181), std::make_pair(151, 83), std::make_pair(179, 151), std::make_pair(144, 178), std::make_pair(190, 120), std::make_pair(100, 129), std::make_pair(78, 31), std::make_pair(119, 176), std::make_pair(33, 201), std::make_pair(92, 27), std::make_pair(43, 65), std::make_pair(38, 102), std::make_pair(47, 218), std::make_pair(5, 180), std::make_pair(13, 73), std::make_pair(56, 200), std::make_pair(63, 189), std::make_pair(251, 70), std::make_pair(81, 247), std::make_pair(167, 49), std::make_pair(104, 233), std::make_pair(3, 220), std::make_pair(180, 43), std::make_pair(205, 68), std::make_pair(46, 82), std::make_pair(204, 203), std::make_pair(164, 144), std::make_pair(143, 143), std::make_pair(14, 253), std::make_pair(250, 168), std::make_pair(230, 227), std::make_pair(59, 32), std::make_pair(194, 42), std::make_pair(148, 53), std::make_pair(49, 170), std::make_pair(19, 20), std::make_pair(145, 111), std::make_pair(120, 48), std::make_pair(83, 103), std::make_pair(4, 169), std::make_pair(211, 30), std::make_pair(168, 57), std::make_pair(112, 5), std::make_pair(44, 4), std::make_pair(31, 198), std::make_pair(242, 156), std::make_pair(231, 92), std::make_pair(202, 115), std::make_pair(53, 14), std::make_pair(181, 158), std::make_pair(245, 160), std::make_pair(9, 204), std::make_pair(131, 195), std::make_pair(209, 17), std::make_pair(51, 217), std::make_pair(123, 89), std::make_pair(12, 62), std::make_pair(207, 225), std::make_pair(212, 7), std::make_pair(208, 93), std::make_pair(200, 60), std::make_pair(54, 221), std::make_pair(72, 66), std::make_pair(163, 172), std::make_pair(221, 51), std::make_pair(253, 141), std::make_pair(158, 140), std::make_pair(0, 77), std::make_pair(106, 104), std::make_pair(157, 210), std::make_pair(125, 123), std::make_pair(29, 132), std::make_pair(214, 38), std::make_pair(116, 22), std::make_pair(247, 74), std::make_pair(16, 110), std::make_pair(42, 249), std::make_pair(15, 150), std::make_pair(134, 175), std::make_pair(222, 238), std::make_pair(215, 194), std::make_pair(169, 63), std::make_pair(173, 149), std::make_pair(170, 177), std::make_pair(183, 224), std::make_pair(27, 223), std::make_pair(124, 55), std::make_pair(139, 229), std::make_pair(61, 108), std::make_pair(153, 171), std::make_pair(238, 237), std::make_pair(70, 40), std::make_pair(255, 163), std::make_pair(8, 136), std::make_pair(189, 162), std::make_pair(91, 58), std::make_pair(40, 241), std::make_pair(36, 37), std::make_pair(132, 138), std::make_pair(239, 119), std::make_pair(34, 155), std::make_pair(152, 124), std::make_pair(75, 153), std::make_pair(87, 164), std::make_pair(232, 146), std::make_pair(60, 235), std::make_pair(45, 45), std::make_pair(10, 47), std::make_pair(166, 228), std::make_pair(137, 75)};
    std::shuffle(characters.begin(), characters.end(), generator);

    for (int i = 0; i < UNIQUE_CHARACTERS; i++)
    {
        originalCharacter[i] = characters[i].first;
        substitutedCharacter[i] = characters[i].second;
    }

    for (int i = 0; i < STRING_LEN; i++){
        decryptedMessage[i] = generator()/255;
    }
}

#endif // UTILITY_H
