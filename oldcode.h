#ifndef OLDCODE_H
#define OLDCODE_H
#include "matmul.h"
matrix matmul_old(matrix mat0, matrix mat1){
	matrix result;
	if(mat0.col!=mat1.row){
		result.row = 0;
		result.col = 0;
		result.mat = nullptr;
		return result;
	}
	result.row = mat0.row;
	result.col = mat1.col;
	result.mat = (double*)malloc(mat0.row*mat1.col*sizeof(double));

	for(uint64_t i=0; i<mat0.row; ++i){
		for(uint64_t j=0; j<mat1.col; ++j){
			double sum = 0;
			for(uint64_t k=0; k<mat0.col; ++k){
				sum += mat0.mat[i*mat0.col+k]*mat1.mat[k*mat1.col+j];
			}
			result.mat[i*result.col+j] = sum;
		}
	}
	
	return result;
}

float exp(float x){
	const float log2e = 1.44269504088896340736;
	const float ln2c1 = 0.693359375f;
	const float ln2c2 = -2.12194440e-4f;

	float x_safe = (x > 88.722839f) ? 88.722839f : ((x < -87.33654f) ? -87.33654f : x); 
	double z = (double)x_safe*log2e;
	int n = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));

	float r = x_safe-n*ln2c1;
	r -= n*ln2c2;

	//float er = 1+r*(1+r*(0.5000001201+r*(0.1666665459+r*(0.0416657959+r*0.0083334519))));
	float er = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.04166667f + r * (0.00833333f + r * (0.00138889f + r * 0.00019841f))))));
	union {
        float f;
        uint32_t i;
    } u;

    u.f = er;
	int32_t raw_exp = (int32_t)((u.i >> 23) & 0xFF) + n;

    uint32_t mask_overflow = -((uint32_t)(raw_exp >= 255));
    uint32_t mask_underflow = -((uint32_t)(raw_exp <= 0));
    
    int32_t clamped_exp = (raw_exp & (int32_t)(~mask_overflow & ~mask_underflow)) 
                         | (255 & mask_overflow);

    u.i = (u.i & 0x807FFFFF & ~mask_overflow) | (((uint32_t)clamped_exp << 23) & ~mask_underflow);

    return u.f;
}

/*	
	matrix smat0;
	matrix smat1;

	smat0.row = 1024;
	smat0.col = 1024;
	smat0.mat = (float*)calloc(smat0.col*smat0.row,sizeof(float));

	smat1.row = 1024;
	smat1.col = 1024;
	smat1.mat = (float*)calloc(smat1.col*smat1.row,sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	matrix smat2 = matmul(smat0, smat1);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    );
	//print_matrix(smat2);
	std::cout << "Time: " << elapsed.count() << " ms\n";

	free(smat0.mat);
	free(smat1.mat);
	free(smat2.mat);
	*/

#endif