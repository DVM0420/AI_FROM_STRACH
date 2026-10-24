#ifndef EXP_H
#define EXP_H

static inline float pow2_int(int k) {
    uint32_t bits = (uint32_t)(k + 127) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}


inline float _software_exp(float x) {
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

#endif