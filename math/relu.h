#ifndef RELU_H
#define RELU_H

inline void reluf(size_t size, float* arr){
	for(uint64_t i=0; i<size;++i){arr[i]=arr[i]>0.0f?arr[i]:0.0f;}
}

inline float _software_reluf(float a){
	return a>0.0f?a:0.0f;
}
void _sse2_array_reluf(size_t size, const float* in, float* out){
	size_t idx = 0;
	__m128 zeros = _mm_setzero_ps();
 
	while(size>=8){
		__m128 v0 = _mm_loadu_ps(&in[idx + 0]);
		__m128 v1 = _mm_loadu_ps(&in[idx + 4]);
 
		_mm_storeu_ps(&out[idx + 0], _mm_max_ps(v0, zeros));
		_mm_storeu_ps(&out[idx + 4], _mm_max_ps(v1, zeros));
 
		size -= 8;
		idx  += 8;
	}
	while(size>=4){
		__m128 vec = _mm_loadu_ps(&in[idx]);
		_mm_storeu_ps(&out[idx], _mm_max_ps(vec, zeros));
		size -= 4;
		idx  += 4;
	}
	while(size>0){
		__m128 fin = _mm_load_ss(&in[idx]);
		_mm_store_ss(&out[idx], _mm_max_ss(fin, zeros));
		--size;
		++idx;
	}
}
 
void _avx_array_reluf(size_t size, const float* in, float* out){
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
		__m128 fin = _mm_load_ss(&in[idx]);
		_mm_store_ss(&out[idx], _mm_max_ss(fin, _mm256_castps256_ps128(zeros)));
		--size;
		++idx;
	}
}
 
void _sse2_inplace_array_reluf(size_t size, float* arr){
	size_t idx = 0;
	__m128 zeros = _mm_setzero_ps();
 
	while(size>=8){
		__m128 v0 = _mm_loadu_ps(&arr[idx + 0]);
		__m128 v1 = _mm_loadu_ps(&arr[idx + 4]);
 
		_mm_storeu_ps(&arr[idx + 0], _mm_max_ps(v0, zeros));
		_mm_storeu_ps(&arr[idx + 4], _mm_max_ps(v1, zeros));
 
		size -= 8;
		idx  += 8;
	}
	while(size>=4){
		__m128 vec = _mm_loadu_ps(&arr[idx]);
		_mm_storeu_ps(&arr[idx], _mm_max_ps(vec, zeros));
		size -= 4;
		idx  += 4;
	}
	while(size>0){
		__m128 fin = _mm_load_ss(&arr[idx]);
		_mm_store_ss(&arr[idx], _mm_max_ss(fin, zeros));
		--size;
		++idx;
	}
}
 
void _avx_inplace_array_reluf(size_t size, float* arr){
	size_t idx = 0;
	__m256 zeros = _mm256_setzero_ps();
 
	while(size>=16){
		__m256 v0 = _mm256_loadu_ps(&arr[idx + 0]);
		__m256 v1 = _mm256_loadu_ps(&arr[idx + 8]);
 
		_mm256_storeu_ps(&arr[idx + 0], _mm256_max_ps(v0, zeros));
		_mm256_storeu_ps(&arr[idx + 8], _mm256_max_ps(v1, zeros));
 
		size -= 16;
		idx  += 16;
	}
	while(size>=8){
		__m256 vec = _mm256_loadu_ps(&arr[idx]);
		_mm256_storeu_ps(&arr[idx], _mm256_max_ps(vec, zeros));
		size -= 8;
		idx  += 8;
	}
	while(size>0){
		__m128 fin = _mm_load_ss(&arr[idx]);
		_mm_store_ss(&arr[idx], _mm_max_ss(fin, _mm256_castps256_ps128(zeros)));
		--size;
		++idx;
	}
}

#endif