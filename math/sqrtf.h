#ifndef SQRTF_H
#define SQRTF_H
#include "../cpplib.h"

void _sse2_array_sqrtf(size_t size, const float* in, float* out){
	size_t idx = 0;
	while(size >= 8){
		__m128 v0 = _mm_loadu_ps(&in[idx + 0]);
		__m128 v1 = _mm_loadu_ps(&in[idx + 4]);

		_mm_storeu_ps(&out[idx + 0], _mm_sqrt_ps(v0));
		_mm_storeu_ps(&out[idx + 4], _mm_sqrt_ps(v1));

		size -= 8;
		idx  += 8;
	}
	while(size >= 4){
		__m128 vec = _mm_loadu_ps(&in[idx]);
		_mm_storeu_ps(&out[idx], _mm_sqrt_ps(vec));
		size -= 4;
		idx  += 4;
	}
	while(size > 0){
		__m128 v = _mm_load_ss(&in[idx]);
		_mm_store_ss(&out[idx], _mm_sqrt_ss(v));
		--size;
		++idx;
	}
}

void _sse2_inplace_array_sqrtf(size_t size, float* arr){
	size_t idx = 0;
	while(size >= 8){
		__m128 v0 = _mm_loadu_ps(&arr[idx + 0]);
		__m128 v1 = _mm_loadu_ps(&arr[idx + 4]);
		_mm_storeu_ps(&arr[idx + 0], _mm_sqrt_ps(v0));
		_mm_storeu_ps(&arr[idx + 4], _mm_sqrt_ps(v1));
		size -= 8;
		idx  += 8;
	}
	while(size >= 4){
		__m128 vec = _mm_loadu_ps(&arr[idx]);
		_mm_storeu_ps(&arr[idx], _mm_sqrt_ps(vec));
		size -= 4;
		idx  += 4;
	}
	while(size > 0){
		__m128 v = _mm_load_ss(&arr[idx]);
		_mm_store_ss(&arr[idx], _mm_sqrt_ss(v));
		--size;
		++idx;
	}
}
void _avx2_array_sqrtf(size_t size, const float* in, float* out){
	size_t idx = 0;
	while(size>=16){
		__m256 v0 = _mm256_loadu_ps(&in[idx + 0]);
		__m256 v1 = _mm256_loadu_ps(&in[idx + 8]);

		_mm256_storeu_ps(&out[idx + 0], _mm256_sqrt_ps(v0));
		_mm256_storeu_ps(&out[idx + 8], _mm256_sqrt_ps(v1));

		size -= 16;
		idx  += 16;
	}
	while(size>=8){
		__m256 vec = _mm256_loadu_ps(&in[idx]);
		_mm256_storeu_ps(&out[idx], _mm256_sqrt_ps(vec));
		size -= 8;
		idx  += 8;
	}
	while(size>0){
		__m128 v = _mm_load_ss(&in[idx]);
		_mm_store_ss(&out[idx], _mm_sqrt_ss(v));
		--size;
		++idx;
	}
}
void _avx2_inplace_array_sqrtf(size_t size, float* arr){
	size_t idx = 0;
	while(size>=16){
		__m256 v0 = _mm256_loadu_ps(&arr[idx + 0]);
		__m256 v1 = _mm256_loadu_ps(&arr[idx + 8]);

		_mm256_storeu_ps(&arr[idx + 0], _mm256_sqrt_ps(v0));
		_mm256_storeu_ps(&arr[idx + 8], _mm256_sqrt_ps(v1));

		size -= 16;
		idx  += 16;
	}
	while(size>=8){
		__m256 vec = _mm256_loadu_ps(&arr[idx]);
		_mm256_storeu_ps(&arr[idx], _mm256_sqrt_ps(vec));
		size -= 8;
		idx  += 8;
	}
	while(size>0){
		__m128 v = _mm_load_ss(&arr[idx]);
		_mm_store_ss(&arr[idx], _mm_sqrt_ss(v));
		--size;
		++idx;
	}
}


float _software_sqrtf(float x)
{
	static const float rsq_tab[128] = {
		0.99611649f, 0.988483301f, 0.981022943f, 0.973728991f,
		0.966595349f, 0.95961623f, 0.952786135f, 0.946099834f,
		0.939552351f, 0.93313895f, 0.926855114f, 0.92069654f,
		0.914659121f, 0.908738935f, 0.902932237f, 0.897235447f,
		0.89164514f, 0.886158041f, 0.880771012f, 0.875481048f,
		0.870285269f, 0.865180913f, 0.860165329f, 0.855235974f,
		0.850390406f, 0.845626276f, 0.84094133f, 0.836333398f,
		0.831800392f, 0.827340304f, 0.8229512f, 0.818631216f,
		0.814378558f, 0.810191494f, 0.806068355f, 0.802007531f,
		0.798007469f, 0.794066667f, 0.790183677f, 0.786357099f,
		0.782585581f, 0.778867814f, 0.775202533f, 0.771588515f,
		0.768024577f, 0.764509572f, 0.761042391f, 0.757621958f,
		0.754247233f, 0.750917207f, 0.747630902f, 0.744387368f,
		0.741185687f, 0.738024966f, 0.734904339f, 0.731822966f,
		0.72878003f, 0.725774739f, 0.722806322f, 0.719874033f,
		0.716977144f, 0.714114948f, 0.711286759f, 0.708491908f,
		0.704360725f, 0.698963245f, 0.693687976f, 0.688530373f,
		0.683486126f, 0.678551144f, 0.673721537f, 0.668993608f,
		0.664363839f, 0.659828879f, 0.655385536f, 0.651030767f,
		0.646761667f, 0.642575463f, 0.638469508f, 0.634441269f,
		0.630488325f, 0.62660836f, 0.622799155f, 0.619058586f,
		0.615384615f, 0.61177529f, 0.608228737f, 0.604743157f,
		0.601316822f, 0.597948074f, 0.594635317f, 0.591377017f,
		0.588171698f, 0.585017939f, 0.581914374f, 0.578859684f,
		0.575852601f, 0.572891899f, 0.5699764f, 0.567104964f,
		0.564276493f, 0.561489925f, 0.558744237f, 0.556038437f,
		0.553371571f, 0.550742713f, 0.548150968f, 0.545595472f,
		0.543075387f, 0.540589903f, 0.538138235f, 0.535719624f,
		0.533333333f, 0.530978649f, 0.52865488f, 0.526361356f,
		0.524097426f, 0.521862458f, 0.519655842f, 0.517476982f,
		0.515325301f, 0.513200239f, 0.511101252f, 0.50902781f,
		0.5069794f, 0.504955522f, 0.502955691f, 0.500979433f,
	};

	uint32_t ix, mant, odd, idx, fb, sb;
	int32_t  E, k;
	float    f, y, h, s, e;

	memcpy(&ix, &x, 4);
	if (ix - 0x00800000u >= 0x7f000000u) {              /* not a positive normal */
		if (ix == 0u || ix == 0x80000000u) return x;     /* +-0 */
		if (ix > 0x80000000u) return (x - x) / (x - x);  /* negative -> NaN */
		if (ix >= 0x7f800000u) return x + x;             /* +inf or NaN */
		x *= 16777216.0f;                                /* subnormal: scale by 2^24 */
		memcpy(&ix, &x, 4);
		k = -12;
	} else {
		k = 0;
	}

	E    = (int32_t)(ix >> 23) - 127;
	mant = ix & 0x007fffffu;
	odd  = (uint32_t)E & 1u;                 /* make the exponent even */
	idx  = (odd << 6) | (mant >> 17);
	fb   = 0x3f800000u + (odd << 23) + mant; /* f = 1.mant, or 2.mant if odd */
	memcpy(&f, &fb, 4);

	y = rsq_tab[idx];                        /* ~2^-8 relative error */
	h = 0.5f * f;
	y = y * (1.5f - h * y * y);              /* Newton 1: ~2^-15 */
	y = y * (1.5f - h * y * y);              /* Newton 2: ~2^-24 */
	double sd = (double)f * (double)y;
	double ed = (double)f - sd * sd;
	sd += ed * (0.5 * (double)y);
	s = (float)sd;

	k += (E - (int32_t)odd) >> 1;            /* result exponent */
	memcpy(&sb, &s, 4);
	sb += (uint32_t)k << 23;
	memcpy(&s, &sb, 4);
	return s;
}

void _software_array_sqrtf(size_t size, float* in, float* out)
{
    for (float *src = in, *dst = out, *end = in + size; src < end; ++src, ++dst) {
        *dst = _software_sqrtf(*src);
    }
}

#endif