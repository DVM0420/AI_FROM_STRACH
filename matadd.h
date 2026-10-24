#ifndef MATADD_H
#define MATADD_H

#include "datatype.h"
#include "cpplib.h"

int32_t matadd(const matrix* mat0, const matrix* mat1, matrix* out){
	if(mat0==nullptr||mat1==nullptr||out==nullptr){return NULL_POINTER;}
	if(mat0->mat==nullptr||mat1->mat==nullptr||out->mat==nullptr){return NULL_POINTER;}
	if(mat0->row==0||mat0->col==0||mat1->row==0||mat1->col==0){return MATRIX_IS_EMPTY;}
	if(mat0->row!=mat1->row||mat0->col!=mat1->col){return DIMENSION_MISMATCH;}

	uint64_t n;
	if(__builtin_mul_overflow(mat0->row, mat0->col, &n)){return SIZE_OVERFLOW;}

	// Kiểm tra aliasing TRƯỚC khi ghi vào out
	if(ranges_overlap(out->mat, n, mat0->mat, n) ||
	   ranges_overlap(out->mat, n, mat1->mat, n)){
		return OUTPUT_ALIASES_INPUT;
	}

	out->row = mat0->row;
	out->col = mat0->col;

	const float* __restrict matrix0 = mat0->mat;
	const float* __restrict matrix1 = mat1->mat;
	float* __restrict result = out->mat;

	for(uint64_t i=0; i<n; ++i){
		result[i] = matrix0[i] + matrix1[i];
	}
	return 0;
}
#endif