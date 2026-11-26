/* Float32 E4M3 errors with a deterministic, histogram-weighted float64 sum. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static float e4m3(unsigned byte)
{
    unsigned exponent = byte >> 3, fraction = byte & 7;
    return exponent ? ldexpf(1.f + fraction / 8.f, (int)exponent - 7)
                    : fraction / 512.f;
}

static float quantize(float x)
{
    if (x >= 448.f)
        return 448.f;
    int exponent;
    frexpf(x, &exponent);
    float step = ldexpf(1.f, exponent > -5 ? exponent - 4 : -9);
    return nearbyintf(x / step) * step;
}

void fit_rows(const uint8_t *packed, const uint8_t *scales, int rows, int width,
              int threads, uint8_t *indices)
{
    // E2M1 times E4M3 is exactly representable in BF16. Retain magnitudes
    // only: E4M3 rounding and the squared error are sign symmetric.
    const float fp4[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    float values[128 * 8] = {0};
    for (unsigned block = 0; block < 127; ++block)
        for (unsigned code = 0; code < 8; ++code)
            values[8 * block + code] = e4m3(block) * fp4[code];

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int row = 0; row < rows; ++row) {
        int count[128 * 8] = {0};
        const uint8_t *codes = packed + (size_t)row * (width / 2);
        const uint8_t *blocks = scales + (size_t)row * (width / 16);
        for (int block = 0; block < width / 16; ++block) {
            int *bucket = count + 8 * (blocks[block] & 127);
            for (int i = 0; i < 8; ++i) {
                unsigned byte = codes[8 * block + i];
                ++bucket[byte & 7];
                ++bucket[byte >> 4 & 7];
            }
        }
        float maximum = 1e-30f;
        int used[128 * 8], length = 0;
        for (int i = 0; i < 128 * 8; ++i)
            if (count[i] && values[i]) {
                maximum = fmaxf(maximum, values[i]);
                used[length++] = i;
            }

        double best = INFINITY;
        for (int j = 0; j < 41; ++j) {
            float scale = maximum / 256.f * (0.5f + j / 32.f);
            double error = 0;
            for (int i = 0; i < length; ++i) {
                int code = used[i];
                float w = values[code];
                float delta = quantize(w / scale) * scale - w;
                float square = delta * delta;
                error += (double)square * count[code];
            }
            if (error < best) {
                best = error;
                indices[row] = j;
            }
        }
    }
}
