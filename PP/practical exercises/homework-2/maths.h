#include <cmath>
#include <random>
#define PI 3.1415926535897932385f

struct Color {
    int r, g, b;
    Color() : r{0}, g{0}, b{0} {}
    Color(int rr, int gg, int bb) : r(rr), g(gg), b(bb) {}
};

struct Vector3 {
    float x, y, z;

    Vector3() : x{0}, y{0}, z{0} {}
    Vector3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}

    Vector3 operator-() const { return Vector3(-this->x, -this->y, -this->z); }

    Vector3 operator+(const Vector3& v) const { return Vector3(this->x + v.x, this->y + v.y, this->z + v.z); }

    Vector3 operator-(const Vector3& v) const { return Vector3(this->x - v.x, this->y - v.y, this->z - v.z); }

    Vector3 operator*(const float& s) const { return Vector3(this->x * s, this->y * s, this->z * s); }

    Vector3 operator/(const float& s) const { return Vector3(this->x / s, this->y / s, this->z / s); }

    Vector3 operator+=(const Vector3& v) {
        this->x += v.x;
        this->y += v.y;
        this->z += v.z;
        return *this;
    }

    Vector3 operator-=(const Vector3& v) { return *this += -v; }

    Vector3 operator*=(const float& s) {
        this->x *= s;
        this->y *= s;
        this->z *= s;
        return *this;
    }

    Vector3 operator/=(const float& s) { return *this *= 1/s; }
};

struct Checksum {
    uint32_t r, g, b;

    Checksum() : r{0}, g{0}, b{0} {}
    Checksum(uint32_t rr, uint32_t gg, uint32_t bb) : r(rr), g(gg), b(bb) {}

    Checksum operator+(const Checksum& v) const { return Checksum(this->r + v.r, this->b + v.b, this->g + v.g); }

    Checksum operator+=(const Checksum& v) {
        this->r += v.r;
        this->g += v.g;
        this->b += v.b;
        return *this;
    }
};

inline float dot(const Vector3& a, const Vector3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vector3 cross(const Vector3& a, const Vector3& b) { return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline float length(const Vector3& vec3) { return std::sqrt(vec3.x * vec3.x + vec3.y * vec3.y + vec3.z * vec3.z); }
inline Vector3 unit_vector(const Vector3& vec3) { return vec3 / length(vec3); }
inline Vector3 reflect(const Vector3 vec3, const Vector3 normal) { return vec3 - normal * 2 * dot(vec3, normal); }

struct Ray {
    Vector3 origin_point;
    Vector3 direction;

    Ray() {}
    Ray(Vector3 origin, Vector3 dir) : origin_point(origin), direction(dir) {}
};

inline Vector3 ray_at(const Ray& ray, float t) { return ray.origin_point + ray.direction * t; }

inline int clamp(int num, int min, int max) {
    int temp1 = num > min ? num : min;
    int temp2 = temp1 < max ? temp1 : max;
    return temp2;
}

inline float clamp(float num, float min, float max) {
    float temp1 = num > min ? num : min;
    float temp2 = temp1 < max ? temp1 : max;
    return temp2;
}

inline float random_float_srand() { return rand() / (RAND_MAX + 1.0f); }
inline float random_float_srand(float min, float max) { return min + (max-min) * random_float_srand(); }

float random_float() {
    static thread_local std::mt19937 generator;
    static std::uniform_real_distribution<> distribution(0.0, 1.0);
    return distribution(generator);
}
inline float random_float(float min, float max) { return min + (max-min) * random_float(); }
inline Vector3 random_vector3() { return Vector3(random_float(), random_float(), random_float()); }
inline Vector3 random_vector3(float min, float max) { return Vector3(random_float(min,max), random_float(min,max), random_float(min,max)); }

struct Camera {
    float aspect_ratio;
    float lens_radius;
    Vector3 origin, lower_left_corner, horizontal, vertical;
    Vector3 u, v, w;

    Camera(Vector3 position, Vector3 look_at, Vector3 up, float aspect_r, float vertical_fov, float aperture, float focus_dist) : aspect_ratio(aspect_r) {
        auto theta = vertical_fov * PI / 180.0f;
        auto h = tan(theta/2);
        auto viewport_height = 2.0f * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(position - look_at);
        u = unit_vector(cross(up, w));
        v = unit_vector(cross(w, u));

        origin = position;
        horizontal = u * viewport_width * focus_dist;
        vertical = v * viewport_height * focus_dist;
        lower_left_corner = origin - horizontal/2 - vertical/2 - w * focus_dist;

        lens_radius = aperture / 2;
    }
};

Vector3 random_in_unit_disk() {
    while (true) {
        auto p = Vector3(random_float(-1,1), random_float(-1,1), 0);
        if (dot(p, p) >= 1) continue;
        return p;
    }
}

Ray get_camera_ray(const Camera& cam, float u, float v) { 
    Vector3 rd = random_in_unit_disk() * cam.lens_radius;
    Vector3 offset = cam.u * rd.x + cam.v * rd.y;

    return Ray(cam.origin + offset, cam.lower_left_corner + cam.horizontal * u + cam.vertical * v - cam.origin - offset); 
}