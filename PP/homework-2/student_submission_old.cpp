#include <unistd.h>
#include <string.h>
#include "raytracer.h"
#include <thread>
#include <mutex>

void aggregate_checksum(Checksum& checksum, uint32_t* row_colors, int pos) {
    checksum.r += row_colors[pos];
    checksum.g += row_colors[pos+1];
    checksum.b += row_colors[pos+2];
}

Color compute_color_no_checksum(Vector3 pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;

    // Divide the color by the number of samples.
    auto scale = 1.0 / samples_per_pixel;
    r *= scale;
    g *= scale;
    b *= scale;

    // Write the translated [0,255] value of each color component.
    int pixel_r = static_cast<int>(256 * clamp(r, 0.0, 0.999));
    int pixel_g = static_cast<int>(256 * clamp(g, 0.0, 0.999));
    int pixel_b = static_cast<int>(256 * clamp(b, 0.0, 0.999));

    return {pixel_r, pixel_g, pixel_b};
}

/*
** Checks if the given ray hits a sphere surface and returns.
** Also returns hit data which contains material information.
*/
bool check_sphere_hit(const std::vector<Sphere>& spheres, const Ray& ray, float t_min, float t_max, Hit& hit) {
    Hit closest_hit;
    bool has_hit = false;
    auto closest_hit_distance = t_max;
    Material material;

    for(std::size_t i = 0; i < spheres.size(); i++) {
        const auto& sphere = spheres[i];
        if(sphere_hit(sphere, ray, t_min, closest_hit_distance, closest_hit)) {
            has_hit = true;
            closest_hit_distance = closest_hit.t;
            material = sphere.material;
        }
    }

    if(has_hit) {
        hit = closest_hit;
        hit.material = material;
    }
    
    return has_hit;
}

/*
** Traces a ray, returns color for the corresponding pixel.
*/
Vector3 trace_ray(const Ray& ray, const std::vector<Sphere>& spheres, int depth) {
    if (depth <= 0) {
        return Vector3(0, 0, 0);
    }

    Hit hit;
    if(check_sphere_hit(spheres, ray, 0.002f, FLT_MAX, hit)) {
        Ray outgoing_ray;
        Vector3 attenuation;

        if(metal_scater(hit.material, ray, hit, attenuation, outgoing_ray)) {
            auto ray_color = trace_ray(outgoing_ray, spheres, depth - 1);
            return Vector3(ray_color.x * attenuation.x, ray_color.y * attenuation.y, ray_color.z * attenuation.z);
        }

        return Vector3(0, 0, 0);
    }

    Vector3 unit_direction = unit_vector(ray.direction);
    auto t = 0.5 * (unit_direction.y + 1.0);
    return Vector3(1.0, 1.0, 1.0) * (1.0 - t) + Vector3(0.5, 0.75, 1.0) * t;
}

void job(int y, const int& height, const int& width, int* image_data, uint32_t* row_colors, const Camera& camera, 
    const std::vector<Sphere>& spheres, const int& depth, const int& samples) {
    uint32_t r = 0, g = 0, b = 0;

    for(int x = 0; x < width; x++) {
        Vector3 pixel_color(0,0,0);
        for(int s = 0; s < samples; s++) {
            auto u = (float) (x + random_float()) / (width - 1);
            auto v = (float) (y + random_float()) / (height - 1);
            auto r = get_camera_ray(camera, u, v);
            pixel_color += trace_ray(r, spheres, depth);
        }

        auto output_color = compute_color_no_checksum(pixel_color, samples);

        r += output_color.r;
        g += output_color.g;
        b += output_color.b;

        int pos = ((height - 1 - y) * width + x) * 3;
        image_data[pos] = output_color.r;
        image_data[pos + 1] = output_color.g;
        image_data[pos + 2] = output_color.b;
    }
    int pos = y * 3;

    row_colors[pos] = r;
    row_colors[pos+1] = g;
    row_colors[pos+2] = b;
}


int main(int argc, char **argv) {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int samples = NUM_SAMPLES;
    int depth = SAMPLE_DEPTH;

    // threading
    int num_threads = std::thread::hardware_concurrency();
    int num_blocks = height / num_threads;
    int num_residuals = height - (num_blocks * num_threads);
    std::thread threads[num_threads];

    // This option parsing is not very interesting.
    int no_output = 0;
    char file_name[256] = "render.ppm";
    int c;
    while ((c = getopt(argc, argv, "d:s:r:n:f:")) != -1)
    {
        switch (c)
        {
            case 'd':
                if (sscanf(optarg, "%d", &depth) != 1)
                    goto error;
                break;
            case 's':
                if (sscanf(optarg, "%d", &samples) != 1)
                    goto error;
                break;
            case 'r':
                if (sscanf(optarg, "%dx%d", &width, &height) != 2)
                    goto error;
                break;
            case 'n':
                if (sscanf(optarg, "%d", &no_output) != 1)
                    goto error;
                break;
            case 'f':
                strncpy(file_name, optarg, sizeof(file_name));
                file_name[255] = '\0'; // safe-guard null-terminator to disable gcc warning
                break;
            case '?':
            error: fprintf(stderr,
                        "Usage:\n"
                        "-d \t number of times a ray can bounce\n"
                        "-s \t number of samples per pixel\n"
                        "-r \t image resolution to be computed\n"
                        "-f \t output file name\n"
                        "-n \t no output(default: 0)\n"
                        "\n"
                        "Example:\n"
                        "%s -d 10 -s 50 -r 720x480 -f tracer.ppm\n",
                        argv[0]);
                exit(EXIT_FAILURE);
                break;
        }
    }

    // Calculating the aspect ratio and creating the camera for the rendering
    const auto aspect_ratio = (float) width / height;
    Camera camera(Vector3(0,1,1), Vector3(0,0,-1), Vector3(0,1,0), aspect_ratio, 90, 0.0f, 1.5f);

    std::vector<Sphere> spheres;

    if (!no_output)
        fprintf(stderr, "Output file: %s\n", file_name);
    else {
        fprintf(stderr, "No output will be written\n");
    }

    readInput();
    create_random_scene(spheres);

    auto image_data = static_cast<int*>(malloc(width * height * sizeof(int) * 3));

    // checksums for each color individually
    Checksum checksum(0, 0, 0);

    auto row_colors = static_cast<uint32_t*>(malloc(height * sizeof(uint32_t) * 3));
    int starting_row = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        int row = starting_row + thread_id;
        threads[thread_id] = std::thread(job, row, std::ref(height), std::ref(width), std::ref(image_data),
            std::ref(row_colors), std::ref(camera), std::ref(spheres), std::ref(depth), std::ref(samples));
    }

    for (int y = 1; y < num_blocks; ++y) {
        starting_row = y * num_threads;
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads[thread_id].join();

            int row = starting_row + thread_id;
            int pos = (row - num_threads) * 3;
            
            aggregate_checksum(checksum, row_colors, pos);

            threads[thread_id] = std::thread(job, row, std::ref(height), std::ref(width), std::ref(image_data),
                std::ref(row_colors), std::ref(camera), std::ref(spheres), std::ref(depth), std::ref(samples));
        }
    }

    starting_row = num_blocks * num_threads;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        threads[thread_id].join();
        
        int row = starting_row + thread_id;
        int pos = (row - num_threads) * 3;

        aggregate_checksum(checksum, row_colors, pos);
        
        if (thread_id < num_residuals) {
            threads[thread_id] = std::thread(job, row, std::ref(height), std::ref(width), std::ref(image_data),
                    std::ref(row_colors), std::ref(camera), std::ref(spheres), std::ref(depth), std::ref(samples));
        }

    }

    for (int thread_id = 0; thread_id < num_residuals; ++thread_id) {
        threads[thread_id].join();

        int row = starting_row + thread_id;
        int pos = (row - num_threads) * 3;

        aggregate_checksum(checksum, row_colors, pos);
    }

    //Saving the render with PPM format
    if(!no_output) {
        FILE* file;
        if ((file = fopen(file_name, "w")) == NULL )
        {
            perror("fopen");
            exit(EXIT_FAILURE);
        }
        if (fprintf(file, "P3\n%d %d %d\n", width, height, 255) < 0)
        {
            perror("fprintf");
            exit(EXIT_FAILURE);
        }
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int pos = (y * width + x) * 3;
                if (fprintf(file, "%d %d %d\n", image_data[pos] , image_data[pos + 1] , image_data[pos + 2] ) < 0)
                {
                    perror("fprintf");
                    exit(EXIT_FAILURE);
                }
            }
        }
        fclose(file);
    }
    writeOutput(checksum);
    free(image_data);
    return 0;
}
