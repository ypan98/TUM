#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <vector>
#include <cfloat>
#include "maths.h"

#define IMAGE_WIDTH 1600
#define IMAGE_HEIGHT 1200
#define NUM_SAMPLES 60
#define SAMPLE_DEPTH 30
#define NUM_SPHERES 12

Color compute_color(Checksum &checksum, Vector3 pixel_color, int samples_per_pixel) {
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

    checksum.r += pixel_r;
    checksum.g += pixel_g;
    checksum.b += pixel_b;

    return {pixel_r, pixel_g, pixel_b};
}

struct Material {
    Vector3 albedo;
    float fuzziness;
};

struct Sphere {
    Vector3 center;
    float radius;
    Material material;

    Sphere(Vector3 cen, float r, Material mat) : center(cen), radius(r), material(mat) {}
};

struct Hit {
    Vector3 point;
    Vector3 normal;
    float t;
    bool front_face;
    Material material;
};

Vector3 random_in_unit_sphere() {
    while (true) {
        auto p = random_vector3(-1,1);
        if (dot(p,p) >= 1) continue;
        return p;
    }
}

bool metal_scater(const Material& material, const Ray& incoming_ray, const Hit& hit, Vector3& attenuation, Ray& outgoing_ray) {
    auto reflected_vec = reflect(unit_vector(incoming_ray.direction), hit.normal);
    outgoing_ray = Ray(hit.point, reflected_vec + random_in_unit_sphere() * material.fuzziness);
    attenuation = material.albedo;
    return (dot(outgoing_ray.direction, hit.normal) > 0);
}

inline void set_face_normal(Hit& hit, const Ray& ray, const Vector3& outward_normal) {
    hit.front_face = dot(ray.direction, outward_normal) < 0;
    hit.normal = hit.front_face ? outward_normal :-outward_normal;
}

bool sphere_hit(const Sphere& sphere, const Ray& ray, float t_min, float t_max, Hit& hit) {
    auto diff = ray.origin_point - sphere.center;

    auto a = dot(ray.direction, ray.direction);
    auto b = dot(diff, ray.direction);
    auto c = dot(diff, diff) - sphere.radius * sphere.radius;
    
    auto discriminant = b*b - a*c;

    if(discriminant > 0) {
        auto discriminant_sqrt = sqrt(discriminant);
        auto first_root = (-b - discriminant_sqrt) / a;

        if(first_root > t_min && first_root < t_max) {
            hit.t = first_root;
            hit.point = ray_at(ray, hit.t);
            auto outward_normal = (hit.point - sphere.center) / sphere.radius;
            set_face_normal(hit, ray, outward_normal);
            return true;
        }

        auto second_root = (-b + discriminant_sqrt) / a;
        if(second_root > t_min && second_root < t_max) {
            hit.t = second_root;
            hit.point = ray_at(ray, hit.t);
            auto outward_normal = (hit.point - sphere.center) / sphere.radius;
            set_face_normal(hit, ray, outward_normal);
            return true;
        }
    }
    return false;
}

void readInput() {
    std::cout << "READY" << std::endl;
    unsigned int seed = 0;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if(seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    // Set the pseudo random number generator seed
    srand(seed);
}

void writeOutput(Checksum checksum) {
    std::cout<< "red checksum is : "<< (double) checksum.r <<std::endl;
    std::cout<< "green checksum is : "<< (double) checksum.g <<std::endl;
    std::cout<< "blue checksum is : "<< (double) checksum.b <<std::endl;

    // This stops the timer.
    std::cout << std::endl << "DONE" << std::endl;
}

void create_random_scene(std::vector<Sphere>& spheres) {
    Material mat_ground;
    mat_ground.albedo = Vector3(0.5, 0.5, 0.5);
    mat_ground.fuzziness = 1.0;

    for(int i = 0; i < NUM_SPHERES; i++) {
        Material mat;
        mat.albedo = Vector3(random_float_srand(0, 1), random_float_srand(0, 1), random_float_srand(0, 1));
        mat.fuzziness = random_float_srand(0, 1);
        spheres.push_back(Sphere(Vector3(random_float_srand(-6, 6),0,random_float_srand(-6, 6)), 0.5f, mat));
    }

    spheres.push_back(Sphere(Vector3(0,-100.5f,-1), 100, mat_ground));
}
