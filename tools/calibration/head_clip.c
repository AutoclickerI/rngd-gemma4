#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { WIDTH = 3840, ROWS = 262144, CLIPS = 36 };
static float paley[60][60], signs[WIDTH];

static uint64_t splitmix64(uint64_t *seed)
{
    uint64_t z = (*seed += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void init_rotation(void)
{
    int residue[59] = {0};
    for (int i = 1; i < 59; ++i)
        residue[i * i % 59] = 1;
    for (int i = 0; i < 60; ++i) {
        for (int j = 0; j < 60; ++j) {
            int difference = (j + 59 - i) % 59;
            float core = 0;
            if (i != j)
                core = j == 0 ? -1 : (i == 0 || residue[difference] ? 1 : -1);
            paley[i][j] = core + (i == j);
        }
    }
    uint64_t seed = UINT64_C(0x7461696c6335);
    for (int i = 0; i < 60; ++i) {
        uint64_t word = splitmix64(&seed);
        for (int j = 0; j < 64; ++j)
            signs[i * 64 + j] = (word >> j) & 1 ? -1 : 1;
    }
}

static float from_bf16(uint16_t bits)
{
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static float round_bf16(float value)
{
    uint32_t word;
    memcpy(&word, &value, sizeof(word));
    word += UINT32_C(0x7fff) + ((word >> 16) & 1);
    return from_bf16(word >> 16);
}

static void rotate(float *input, float *output)
{
    for (int i = 0; i < WIDTH; ++i)
        input[i] *= signs[i];
    for (int block = 0; block < WIDTH; block += 64) {
        for (int stride = 1; stride < 64; stride *= 2) {
            for (int i = block; i < block + 64; i += 2 * stride) {
                for (int j = i; j < i + stride; ++j) {
                    float a = input[j], b = input[j + stride];
                    input[j] = a + b;
                    input[j + stride] = a - b;
                }
            }
        }
    }
    memset(output, 0, WIDTH * sizeof(float));
    for (int i = 0; i < 60; ++i)
        for (int j = 0; j < 60; ++j)
            for (int k = 0; k < 64; ++k)
                output[i * 64 + k] += paley[i][j] * input[j * 64 + k];
    float normalization = 1.f / sqrtf((float)WIDTH);
    for (int i = 0; i < WIDTH; ++i)
        output[i] *= normalization;
}

static float odd_code(float value)
{
    float half = value / 2.f;
    if (half >= 7.f)
        return 15.f;
    if (half < -7.f)
        return -15.f;
    int integer = (int)half;
    return (float)(2 * (integer - (half < (float)integer)) + 1);
}

static uint8_t choose_clip(const uint8_t *row, const int8_t *exponents)
{
    float input[WIDTH], rotated[WIDTH];
    for (int i = 0; i < WIDTH; ++i)
        input[i] = ldexpf(from_bf16(row[2*i] | ((uint16_t)row[2*i+1] << 8)), exponents[i]);
    rotate(input, rotated);
    float maximum = 0;
    for (int i = 0; i < WIDTH; ++i)
        maximum = fmaxf(maximum, fabsf(rotated[i]));
    if (maximum == 0)
        return 0;
    float best_error = INFINITY;
    uint8_t best = 0;
    for (int j = 0; j < CLIPS; ++j) {
        float ratio = (float)(30 + 2*j) / 100.f;
        float scale = round_bf16(ratio * maximum / 15.f);
        float error = 0;
        for (int i = 0; i < WIDTH; ++i) {
            float difference = odd_code(rotated[i] / scale) * scale - rotated[i];
            error += difference * difference;
        }
        if (error < best_error) {
            best_error = error;
            best = j;
        }
    }
    return best;
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        fputs("usage: head_clip WEIGHTS EMBEDDING_OFFSET EXPONENTS OUTPUT THREADS\n", stderr);
        return 2;
    }
    size_t offset = strtoull(argv[2], NULL, 10);
    int threads = atoi(argv[5]);
    assert(threads >= 1 && threads <= 24);
    int8_t exponents[WIDTH];
    FILE *diagonal = fopen(argv[3], "rb");
    assert(diagonal);
    assert(fread(exponents, 1, WIDTH, diagonal) == WIDTH && fgetc(diagonal) == EOF);
    fclose(diagonal);

    int fd = open(argv[1], O_RDONLY);
    assert(fd >= 0);
    struct stat st;
    assert(fstat(fd, &st) == 0);
    assert(offset + (size_t)ROWS * 2 * WIDTH <= (size_t)st.st_size);
    const uint8_t *weights = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(weights != MAP_FAILED);
    uint8_t *indices = malloc(ROWS);
    assert(indices);
    init_rotation();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < ROWS; ++i)
        indices[i] = choose_clip(weights + offset + i * 2 * WIDTH, exponents);

    FILE *out = fopen(argv[4], "wb");
    assert(out);
    assert(fwrite(indices, 1, ROWS, out) == ROWS);
    assert(fclose(out) == 0);
    free(indices);
    munmap((void *)weights, st.st_size);
    close(fd);
}
