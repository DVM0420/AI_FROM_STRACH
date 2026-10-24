#ifndef TANH_H
#define TANH_H
#include "exp.h"

inline float _software_tanh(float x){
	if(x<-9.0f)return -1.0f;
	if(x>9.0f)return 1.0f;
	if(x==0.0f)return 0.0f;

	union {
        float f;
        uint32_t i;
    } u;

    u.f=x;
	u.i &= 0x7fffffff;
	if (u.f>=0.625){
		float s = _software_exp(u.f+u.f);
		u.f = 1-2/(s+1);
		if(x<0){u.f = -u.f;}
	}
	else{
		float s=x*x;
		u.f = ((((-5.70498872745e-3*s+2.06390887954e-2)*s-5.37397155531e-2)*s+1.33314422036e-1)*s-3.33332819422e-1)*s*x + x;
	}
	return u.f;
}

#endif