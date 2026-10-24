#ifndef DATATYPE_H
#define	DATATYPE_H

//START MATRIX
#define MATRIX_IS_EMPTY -1
#define COL_AND_ROW_IS_NOT_EQUAL -2
#define NULL_POINTER -3
#define OUTPUT_ALIASES_INPUT -4
#define DIMENSION_MISMATCH -5
#define SIZE_OVERFLOW -6

struct matrix{
	uint64_t row=0;
	uint64_t col=0;
	float* mat;
};
//END MATRIX


#endif