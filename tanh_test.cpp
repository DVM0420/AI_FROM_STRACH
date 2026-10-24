#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <bitset>
#include <immintrin.h>
using namespace std;

//const double ln2c1 = 0.69314718055994530941;


#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
#endif

bool CHECK_AVX2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    int info[4] = {0};

#if defined(_MSC_VER)
    __cpuid(info, 0);
#elif defined(__GNUC__) || defined(__clang__)
    __get_cpuid(0, (unsigned int*)&info[0], (unsigned int*)&info[1], 
                   (unsigned int*)&info[2], (unsigned int*)&info[3]);
#else
    return false; 
#endif

    if (info[0] < 7) {
        return false; 
    }

#if defined(_MSC_VER)
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;

#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
#endif

#endif
    return false;
}


static inline float pow2_int(int k) {
    uint32_t bits = (uint32_t)(k + 127) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

inline float exp(float x) {
    if (x != x) return x;                        // NaN propagates unchanged
    x = (x < -104.0f) ? -104.0f : x;             // exp(-104) rounds to +0
    x = (x >   89.0f) ?   89.0f : x;             // exp(89) overflows to +inf

    // n = round(x / ln2). Adding 1.5*2^23 leaves n in the low mantissa bits
    // (valid for |x*log2e| < 2^22), so n is read as an integer bit pattern and
    // kf = n is recovered by one exact subtraction.
    const float MAGIC = 12582912.0f;             // 1.5 * 2^23
    const float shifted = x * 1.44269504f + MAGIC;
    uint32_t sb;
    std::memcpy(&sb, &shifted, sizeof sb);
    const int32_t n = (int32_t)(sb - 0x4B400000u);
    const float kf = shifted - MAGIC;

    // r = x - n*ln2, ln2 = LN2_HI + LN2_LO. kf*LN2_HI is exact (11-bit LN2_HI),
    // and the two products are independent of each other.
    const float LN2_HI = 0.693359375f;
    const float LN2_LO = -2.12194440e-4f;
    const float r = (x - kf * LN2_HI) - kf * LN2_LO;   // |r| <= ~0.35

    // exp(r) = 1 + r*g(r), g = sum_{k=0..6} r^k/(k+1)!, in Estrin form with
    // z = r*r. The pair terms run in parallel instead of as a serial Horner
    // chain. Keeping the form 1 + r*g leaves one rounding at magnitude 1.
    const float z  = r * r;
    const float z2 = z * z;
    const float a0 = 1.0f + 0.5f * r;                        // 1 + r/2
    const float a1 = (1.0f / 6.0f) + (1.0f / 24.0f) * r;     // 1/6 + r/24
    const float a2 = (1.0f / 120.0f) + (1.0f / 720.0f) * r;  // 1/120 + r/720
    const float g  = (a0 + a1 * z) + z2 * (a2 + (1.0f / 5040.0f) * z);
    const float p  = 1.0f + r * g;

    // Scale by 2^n in two halves so each exponent is representable. The first
    // product is exact; the last multiply does the single correct rounding
    // (including subnormals) and overflows to inf past FLT_MAX.
    const int32_t n1 = n >> 1;
    const int32_t n2 = n - n1;
    return (p * pow2_int(n1)) * pow2_int(n2);
}


float tanh(float x){
	if(x<-9.0f)return -1.0f;
	if(x>9.0f)return 1.0f;
	if(x==0.0f)return 0.0f;

	union {
        float f;
        uint32_t i;
    } u;

    u.f=x;
	u.i &= 0x7fffffff;
	//float p = u.f;

	if (u.f>=0.625){
		float s = exp(u.f+u.f);
		u.f = 1-2/(s+1);
		if(x<0){u.f = -u.f;}
	}
	else{
		float s=x*x;
		u.f = ((((-5.70498872745e-3*s+2.06390887954e-2)*s-5.37397155531e-2)*s+1.33314422036e-1)*s-3.33332819422e-1)*s*x + x;
	}
	return u.f;
}

float fast_sqrtf(float a){
	float x2 = a*0.5f;
	float y = a;

	uint32_t i = *(uint32_t*)&a;
	i = 0x5f3759df-(i>>1);
	y = *(float*)&i;
	y = y*(1.5f-(x2*y*y));
	y = y*(1.5f-(x2*y*y));
	y = y*(1.5f-(x2*y*y)) * a;

	return y;
}

void fast_sqrtf8i(float* a){
	float x2, y;
	uint32_t i;
	for(uint8_t j=0; j<8; ++j){
		x2 = a[j]*0.5f;
		y = a[j];

		i = *(uint32_t*)&a[j];
		i = 0x5f3759df-(i>>1);
		y = *(float*)&i;
		y = y*(1.5f-(x2*y*y));
		y = y*(1.5f-(x2*y*y));
		y = y*(1.5f-(x2*y*y));
		y = y*(1.5f-(x2*y*y));
		a[j] = a[j]*y;
	}

}

float sqrtf_fast(float x)
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
	s = f * y;                               /* sqrt(f) in [1,2] */
	e = f - s * s;                           /* residual */
	s = s + e * (0.5f * y);                  /* Heron correction */

	k += (E - (int32_t)odd) >> 1;            /* result exponent */
	memcpy(&sb, &s, 4);
	sb += (uint32_t)k << 23;
	memcpy(&s, &sb, 4);
	return s;
}

void sqrtf_fast_x8(float* arr)
{
	for (int i = 0; i < 8; ++i) {
		arr[i] = sqrtf_fast(arr[i]);
	}
}

void vsqrtfi(float* arr){//for 8 float numbers
	cout<<"AVX OK\n";
	__m256 v = _mm256_loadu_ps(arr);
	__m256 res = _mm256_sqrt_ps(v);
	_mm256_storeu_ps(arr, res); 

}

void vsqrt_avx256_8(float* arr){//for 64 float numbers
	__m256 v0 = _mm256_loadu_ps(&arr[0]);
    __m256 v1 = _mm256_loadu_ps(&arr[8]);
    __m256 v2 = _mm256_loadu_ps(&arr[16]);
    __m256 v3 = _mm256_loadu_ps(&arr[24]);
    __m256 v4 = _mm256_loadu_ps(&arr[32]);
    __m256 v5 = _mm256_loadu_ps(&arr[40]);
    __m256 v6 = _mm256_loadu_ps(&arr[48]);
    __m256 v7 = _mm256_loadu_ps(&arr[56]);

	v0 = _mm256_sqrt_ps(v0);
	v1 = _mm256_sqrt_ps(v1);
	v2 = _mm256_sqrt_ps(v2);
	v3 = _mm256_sqrt_ps(v3);
	v4 = _mm256_sqrt_ps(v4);
	v5 = _mm256_sqrt_ps(v5);
	v6 = _mm256_sqrt_ps(v6);
	v7 = _mm256_sqrt_ps(v7);

	_mm256_storeu_ps(&arr[0],  v0);
	_mm256_storeu_ps(&arr[8],  v1);
	_mm256_storeu_ps(&arr[16], v2);
	_mm256_storeu_ps(&arr[24], v3);
	_mm256_storeu_ps(&arr[32], v4);
	_mm256_storeu_ps(&arr[40], v5);
	_mm256_storeu_ps(&arr[48], v6);
	_mm256_storeu_ps(&arr[56], v7);

}
void vsqrtfia(float* arr){//for 8 float numbers
	__m256 v = _mm256_load_ps(arr);
	__m256 res = _mm256_sqrt_ps(v);
	_mm256_store_ps(arr, res); 

}

void (*vsqrtf)(float*) = nullptr;

void resolve_function(){
	if(CHECK_AVX2()){
		cout<<"AVX\n";

		vsqrtf = vsqrtfi;
	}
	else{
		vsqrtf = fast_sqrtf8i;
	}
}

int main(){
	resolve_function();

	float a[] = {2,4,6,8,9,12,14,16};
	cout<<"debug0\n";
	cout<<(uint64_t)vsqrtf<<"\n";
	sqrtf_fast_x8(a);
	cout<<"debug1\n";

	for(int i=0; i<8;++i){
		cout<<fixed<<setprecision(15)<<a[i]<<"\n";
	}
	//cout<<fixed<<setprecision(15)<<a;
	return 0;
}



void tanh_256b_ip(float* x){


	//load 8 number into mm256
	__m256 vec = _mm256_loadu_ps(x);
	__m256 zeros = _mm256_setzero_ps();

	__m256 cmp_res = _mm256_cmp_ps(vec, zeros, _CMP_LT_OQ);
	int mask = _mm256_movemask_ps(cmp_res);
	cout<<"mask:"<<std::bitset<8>(mask);
/*	if(x<-9.0f)return -1.0f;
	if(x>9.0f)return 1.0f;
	if(x==0.0f)return 0.0f;

	union {
        float f;
        uint32_t i;
    } u;

    u.f=x;
	u.i &= 0x7fffffff;
	//float p = u.f;

	if (u.f>=0.625){
		float s = exp(u.f+u.f);
		u.f = 1-2/(s+1);
		if(x<0){u.f = -u.f;}
	}
	else{
		float s=x*x;
		u.f = ((((-5.70498872745e-3*s+2.06390887954e-2)*s-5.37397155531e-2)*s+1.33314422036e-1)*s-3.33332819422e-1)*s*x + x;
	}
	return u.f;*/
}


/*const int N = 1000000;
	using clock_t_ = std::chrono::high_resolution_clock;

	// tanh: 1,000,000 điểm cách đều trên [-9, 9]
	const float tmin = -9.0f, tmax = 9.0f;
	const float tstep = (tmax - tmin) / (N - 1);
	double acc_t = 0.0;
	auto t0 = clock_t_::now();
	for (int i = 0; i < N; ++i){
		float x = tmin + i * tstep;
		acc_t += tanh(x);
	}
	auto t1 = clock_t_::now();
	double ms_t = std::chrono::duration<double, std::milli>(t1 - t0).count();

	// exp: 1,000,000 điểm cách đều trên [-87.33654, 88.722839]
	const float emin = -87.33654f, emax = 88.722839f;
	const float estep = (emax - emin) / (N - 1);
	double acc_e = 0.0;
	auto t2 = clock_t_::now();
	for (int i = 0; i < N; ++i){
		float x = emin + i * estep;
		acc_e += exp(x);
	}
	auto t3 = clock_t_::now();
	double ms_e = std::chrono::duration<double, std::milli>(t3 - t2).count();

	cout << fixed << setprecision(6);
	cout << "tanh: " << ms_t << " ms, checksum = " << acc_t << "\n";
	cout << "exp : " << ms_e << " ms, checksum = " << acc_e << "\n";

	float arr[] = {1.0f, 2.0f, 34.0f, 5.0f, -4.0f, 5.0f, 7.0f, -1.0f};
    tanh_256b_ip(arr);
	return 0;*/


/*
int main(){
	float a = 0;
	a = exp(a);
	cout<<fixed<<setprecision(15)<<a;
	//cout<<a;
	return 0;
}
*/