#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>

#include "mandelbrot_set.h"

#define CHANNELS 3
// Returns the one dimensional index into our pseudo 3D array
#define OFFSET(y, x, c) (y * x_resolution * CHANNELS + x * CHANNELS + c)

const double VIEW_X0 = -2;
const double VIEW_X1 = +2;
const double VIEW_Y0 = -2;
const double VIEW_Y1 = +2;

/*
 * TODO@Students: This is your kernel. Take a look at any dependencies and how to parallelize it
 */
int mandelbrot_draw(int x_resolution, int y_resolution, int max_iter,
                    double x_stepsize, double y_stepsize,
                    int palette_shift, unsigned char *img,
                    double power, bool no_output) {
    using namespace std::complex_literals;

    double y;
    double x;

    std::complex<double> Z;
    std::complex<double> C;

    int k;

    int pointsInSetCount = 0;

    // For each pixel in the image
    #pragma omp parallel for schedule(dynamic) private(y, x, Z, C, k) reduction(+: pointsInSetCount) num_threads(32)
    for (int i = 0; i < y_resolution; i++) {
        for (int j = 0; j < x_resolution; j++) {
            y = VIEW_Y1 - i * y_stepsize;
            x = VIEW_X0 + j * x_stepsize;

            Z = 0.0 + 0.0i;
            C = x + y * 1.0i;

            k = 0;

            // Apply the Mandelbrot calculation until the absolute value >= 2 (meaning the calculation will diverge to
            // infinity) or the maximum number of iterations was reached.
            do {
                Z = std::pow(Z, power) + C;
                k++;
            } while (std::abs(Z) < 2 && k < max_iter);

            // If the maximum number of iterations was reached then this point is in the Mandelbrot set and we color it
            // black. Else, it is outside and we color it with a color that corresponds to how many iterations there
            // were before we confirmed the divergence.
            if (k == max_iter) {
                pointsInSetCount++;
            }

            if(!no_output) {
                if (k == max_iter) {
                    memcpy(&img[OFFSET(i, j, 0)], black, 3);
                } else {
                    int index = (k + palette_shift) % (sizeof(colors) / sizeof(colors[0]));
                    memcpy(&img[OFFSET(i, j, 0)], colors[index], 3);
                }
            }
        }
    }

    return pointsInSetCount;
}


int main(int argc, char **argv) {
    double power = 1.0;
    int max_iter = sizeof(colors) / sizeof(colors[0]);
    int x_resolution = 1544;
    int y_resolution = 1377;
    int palette_shift = 0;
    char file_name[256] = "mandelbrot.ppm";
    int no_output = 0;

    double x_stepsize;
    double y_stepsize;

    // This option parsing is not very interesting.
    int c;
    while ((c = getopt(argc, argv, "t:p:i:r:v:n:f:")) != -1) {
        switch (c) {
            case 'p':
                if (sscanf(optarg, "%d", &palette_shift) != 1)
                    goto error;
                break;

            case 'i':
                if (sscanf(optarg, "%d", &max_iter) != 1)
                    goto error;
                break;

            case 'r':
                if (sscanf(optarg, "%dx%d", &x_resolution, &y_resolution) != 2)
                    goto error;
                break;
            case 'n':
                if (sscanf(optarg, "%d", &no_output) != 1)
                    goto error;
                break;
            case 'f':
                strncpy(file_name, optarg, sizeof(file_name) - 1);
                break;

            case '?':
            error:
                fprintf(stderr,
                        "Usage:\n"
                        "-i \t maximum number of iterations per pixel\n"
                        "-r \t image resolution to be computed\n"
                        "-p \t shift palette to change colors\n"
                        "-f \t output file name\n"
                        "-n \t no output(default: 0)\n"
                        "\n"
                        "Example:\n"
                        "%s -r 720x480 -i 5000 -f mandelbrot.ppm\n",
                        argv[0]);
                exit(EXIT_FAILURE);
                break;
        }
    }

    FILE *file = nullptr;
    if (!no_output) {
        if ((file = fopen(file_name, "w")) == NULL) {
            perror("fopen");
            exit(EXIT_FAILURE);
        }
    }

    // This is a pseudo 3d (1d storage) array that can be accessed using the OFFSET Macro.
    auto *image = static_cast<unsigned char *>(malloc(x_resolution * y_resolution * sizeof(unsigned char[3])));


    // Calculate the step size.
    x_stepsize = (VIEW_X1 - VIEW_X0) / (1.0 * x_resolution);
    y_stepsize = (VIEW_Y1 - VIEW_Y0) / (1.0 * y_resolution);

    if(x_stepsize <= 0 || y_stepsize <= 0) {
        printf("Please specify a valid range.\n");
        exit(-1);
    }

    // Don't modify anything that prints to stdout.
    // The output on standard out needs to be the same as in the sequential implementation.
    fprintf(stdout,
            "Following settings are used for computation:\n"
            "Max. iterations: %d\n"
            "Resolution: %dx%d\n"
            "View frame: [%lf,%lf]x[%lf,%lf]\n"
            "Stepsize x = %lf y = %lf\n", 
            max_iter, x_resolution, y_resolution,
            VIEW_X0, VIEW_X1, VIEW_Y0, VIEW_Y1,
            x_stepsize, y_stepsize);

    if (!no_output)
        fprintf(stderr, "Output file: %s\n", file_name);
    else
        fprintf(stderr, "No output will be written\n");

    getProblemFromInput(&power);

    // compute mandelbrot
    int numSamplesInSet = mandelbrot_draw(x_resolution, y_resolution, max_iter,
                                          x_stepsize, y_stepsize, palette_shift, image, power, no_output);
    outputSolution(numSamplesInSet);

    if (!no_output) {
        if (fprintf(file, "P6\n%d %d\n%d\n", x_resolution, y_resolution, 255) < 0) {
            std::perror("fprintf");
            exit(EXIT_FAILURE);
        }
        size_t bytes_written = fwrite(image, 1,
                                      x_resolution * y_resolution * sizeof(unsigned char[3]), file);
        if (bytes_written < x_resolution * y_resolution * sizeof(char[3])) {
            std::perror("fwrite");
            exit(EXIT_FAILURE);
        }

        fclose(file);
    }

    free(image);

    return 0;
}
