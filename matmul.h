#ifndef MATMUL_H
#define MATMUL_H
#include "datatype.h"
#include "cpplib.h"

static inline bool ranges_overlap(const float* a, uint64_t na, const float* b, uint64_t nb){
	const uintptr_t a0 = (uintptr_t)a, a1 = (uintptr_t)(a + na);
	const uintptr_t b0 = (uintptr_t)b, b1 = (uintptr_t)(b + nb);
	return a0 < b1 && b0 < a1;
}

int32_t matmul(const matrix* mat0, const matrix* mat1, matrix* out){
	if(mat0==nullptr||mat1==nullptr||out==nullptr){return NULL_POINTER;}
	if(mat0->col==0||mat0->row==0||mat1->row==0||mat1->col==0){return MATRIX_IS_EMPTY;}
	if(mat0->col!=mat1->row){return COL_AND_ROW_IS_NOT_EQUAL;}
	if(mat0->mat==nullptr||mat1->mat==nullptr||out->mat==nullptr){return NULL_POINTER;}

	const uint64_t M = mat0->row, K = mat0->col, N = mat1->col;

	if(ranges_overlap(out->mat, M*N, mat0->mat, M*K) ||
	   ranges_overlap(out->mat, M*N, mat1->mat, K*N)){
		return OUTPUT_ALIASES_INPUT;
	}

	out->row = M;
	out->col = N;
	const float* matrix0 = mat0->mat;
	const float* matrix1 = mat1->mat;
	float* result = out->mat;

	for(uint64_t i=0; i<M; ++i){
		float* __restrict ci = result + i*N;
		const float* ai = matrix0 + i*K;

		{
			const float r = ai[0];
			const float* __restrict b0 = matrix1;
			for(uint64_t j=0; j<N; ++j) ci[j] = r * b0[j];
		}
		for(uint64_t k=1; k<K; ++k){
			const float r = ai[k];
			const float* __restrict bk = matrix1 + k*N;
			for(uint64_t j=0; j<N; ++j) ci[j] += r * bk[j];
		}
	}
	return 0;
}
#endif