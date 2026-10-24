#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#ifdef _MSC_VER
#include <malloc.h>      /* _mm_malloc / _mm_free on MSVC */
#endif

#ifdef _WIN32
#include <windows.h>     /* QueryPerformanceCounter */
#else
#include <time.h>        /* clock_gettime */
#endif

static inline float _software_relu(float x){ return x > 0.0f ? x : 0.0f; }

void relu_v0(size_t size, const float* in, float* out){
    size_t idx = 0;
    __m256 zeros = _mm256_setzero_ps();
    while(size>=16){
        __m256 v0 = _mm256_loadu_ps(&in[idx + 0]);
        __m256 v1 = _mm256_loadu_ps(&in[idx + 8]);
        _mm256_storeu_ps(&out[idx + 0], _mm256_max_ps(v0, zeros));
        _mm256_storeu_ps(&out[idx + 8], _mm256_max_ps(v1, zeros));
        size -= 16;
        idx  += 16;
    }
    while(size>=8){
        __m256 vec = _mm256_loadu_ps(&in[idx]);
        _mm256_storeu_ps(&out[idx], _mm256_max_ps(vec, zeros));
        size -= 8;
        idx  += 8;
    }
    while(size>0){
        out[idx] = _software_relu(in[idx]);
        --size;
        ++idx;
    }
}

void relu_v1(size_t size, const float* in, float* out){
    size_t idx = 0;
    __m256 zeros = _mm256_setzero_ps();
    while(size>=64){
        __m256 v0 = _mm256_loadu_ps(&in[idx + 0]);
        __m256 v1 = _mm256_loadu_ps(&in[idx + 8]);
        __m256 v2 = _mm256_loadu_ps(&in[idx + 16]);
        __m256 v3 = _mm256_loadu_ps(&in[idx + 24]);
        __m256 v4 = _mm256_loadu_ps(&in[idx + 32]);
        __m256 v5 = _mm256_loadu_ps(&in[idx + 40]);
        __m256 v6 = _mm256_loadu_ps(&in[idx + 48]);
        __m256 v7 = _mm256_loadu_ps(&in[idx + 56]);
        _mm256_storeu_ps(&out[idx + 0],  _mm256_max_ps(v0, zeros));
        _mm256_storeu_ps(&out[idx + 8],  _mm256_max_ps(v1, zeros));
        _mm256_storeu_ps(&out[idx + 16], _mm256_max_ps(v2, zeros));
        _mm256_storeu_ps(&out[idx + 24], _mm256_max_ps(v3, zeros));
        _mm256_storeu_ps(&out[idx + 32], _mm256_max_ps(v4, zeros));
        _mm256_storeu_ps(&out[idx + 40], _mm256_max_ps(v5, zeros));
        _mm256_storeu_ps(&out[idx + 48], _mm256_max_ps(v6, zeros));
        _mm256_storeu_ps(&out[idx + 56], _mm256_max_ps(v7, zeros));
        size -= 64;
        idx  += 64;
    }
    while(size>=8){
        __m256 vec = _mm256_loadu_ps(&in[idx]);
        _mm256_storeu_ps(&out[idx], _mm256_max_ps(vec, zeros));
        size -= 8;
        idx  += 8;
    }
    while(size>0){
        out[idx] = _software_relu(in[idx]);
        --size;
        ++idx;
    }
}

static double now_ns(void){
#ifdef _WIN32
    static LARGE_INTEGER freq = { 0 };
    LARGE_INTEGER c;
    if(freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
}

typedef void (*relu_fn)(size_t, const float*, float*);

static double bench(relu_fn fn, size_t n, const float* in, float* out, int iters){
    /* warmup */
    for(int i = 0; i < 3; i++) fn(n, in, out);
    double best = 1e30;
    for(int r = 0; r < 7; r++){
        double t0 = now_ns();
        for(int i = 0; i < iters; i++) fn(n, in, out);
        double t = (now_ns() - t0) / iters;
        if(t < best) best = t;
    }
    return best;
}

int main(void){
    size_t sizes[] = { 1000, 100000, 1000000, 16 * 1024 * 1024 };
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    /* correctness check against scalar reference */
    {
        size_t n = 1003;
        float *in = (float*)_mm_malloc(n*sizeof(float), 32);
        float *o0 = (float*)_mm_malloc(n*sizeof(float), 32);
        float *o1 = (float*)_mm_malloc(n*sizeof(float), 32);
        for(size_t i = 0; i < n; i++) in[i] = (float)((int)(i*2654435761u % 2001) - 1000) / 37.0f;
        relu_v0(n, in, o0); relu_v1(n, in, o1);
        int ok = 1;
        for(size_t i = 0; i < n; i++){
            float ref = _software_relu(in[i]);
            if(o0[i] != ref || o1[i] != ref){ ok = 0; break; }
        }
        printf("correctness: %s\n\n", ok ? "PASS" : "FAIL");
        _mm_free(in); _mm_free(o0); _mm_free(o1);
    }

    printf("%-12s %-12s %-16s %-16s %-10s\n", "elements", "iters", "v0 (16/iter)", "v1 (64/iter)", "v1 vs v0");
    for(int s = 0; s < nsizes; s++){
        size_t n = sizes[s];
        float *in  = (float*)_mm_malloc(n*sizeof(float), 32);
        float *out = (float*)_mm_malloc(n*sizeof(float), 32);
        for(size_t i = 0; i < n; i++) in[i] = (float)((int)(i*2654435761u % 2001) - 1000) / 37.0f;

        int iters = (int)(2e8 / (double)n);
        if(iters < 1) iters = 1;
        if(iters > 100000) iters = 100000;

        double t0 = bench(relu_v0, n, in, out, iters);
        double t1 = bench(relu_v1, n, in, out, iters);
        printf("%-12zu %-12d %-16.1f %-16.1f %+.1f%%\n",
               n, iters, t0, t1, (t1 - t0) / t0 * 100.0);
        /* ns per call shown; compute throughput too */
        printf("   throughput: v0 %.2f GB/s, v1 %.2f GB/s (read+write)\n",
               (double)n*8.0 / t0, (double)n*8.0 / t1);
        _mm_free(in); _mm_free(out);
    }
    return 0;
}