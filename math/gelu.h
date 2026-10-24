#ifndef GELU_H
#define GELU_H
#include "tanh.h"
inline float _software_gelu(float a){
	a = (a < -10.0f) ? -10.0f : a; 
	return 0.5f*a*(1.0f+_software_tanh(0.7978845608f*(a+ a*a*a*0.044715f)));
}

#endif