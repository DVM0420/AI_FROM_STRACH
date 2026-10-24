// callovh.cpp
// Build: g++ -std=c++17 -O2 -fno-omit-frame-pointer -o callovh.exe callovh.cpp
// Run:   start /wait /affinity 8 callovh.exe      (mask 0x8 = logical CPU 3)
//
// Target ABI: Win64 (MinGW-w64). First integer argument arrives in RCX, not RDI.
// Win64 has no red zone, and RDI, RSI are callee-saved.

#include <windows.h>
#include <x86intrin.h>
#include <cstdint>
#include <cstdio>
#include <functional>

#define NOINLINE __attribute__((noinline))

// Names are prefixed because windows.h defines OPAQUE (wingdi.h) as a macro.
// Opaque value: the optimizer must assume v changed arbitrarily. Emits no instructions.
#define BENCH_VAL(v) asm volatile("" : "+r"(v))
// Opaque object: any state reachable from o must be reloaded. Emits no instructions.
#define BENCH_MEM(o) asm volatile("" : : "g"(&(o)) : "memory")
// Side-effect barrier inside each callee: makes it non-pure so the compiler cannot
// fold, devirtualize, or remove the call. Emits no instructions.
#define CALLEE_BARRIER(x) asm volatile("" : "+r"(x))

// ---------- Callees ------------------------------------------------------------

// (1) Free leaf. Pure call/ret cost plus argument/return moves.
NOINLINE uint64_t free_leaf(uint64_t x) { CALLEE_BARRIER(x); return x; }

// (2) Static member. Same codegen as (1); only name mangling differs.
struct S {
    static NOINLINE uint64_t sfn(uint64_t x) { CALLEE_BARRIER(x); return x; }
};

// (3) Stack frame. The local array's address escapes, so the frame must exist.
NOINLINE uint64_t frame_only(uint64_t x) {
    uint64_t buf[32];
    BENCH_MEM(buf);
    return x;
}

// (4) Callee-saved preservation. Win64 callee-saved set: rbx, rdi, rsi, rbp, r12-r15.
// Listing them as clobbers forces the compiler to push and pop each one.
NOINLINE uint64_t callee_saved(uint64_t x) {
    asm volatile("" ::: "rbx", "rdi", "rsi", "r12", "r13", "r14", "r15");
    CALLEE_BARRIER(x);
    return x;
}

// (5) Virtual dispatch target.
struct Base { virtual uint64_t f(uint64_t x) = 0; };
struct Derived final : Base {
    NOINLINE uint64_t f(uint64_t x) override { CALLEE_BARRIER(x); return x; }
};

// ---------- Timing -------------------------------------------------------------

static inline uint64_t tsc_begin() { _mm_lfence(); return __rdtsc(); }
static inline uint64_t tsc_end() {
    unsigned aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

// Dependency-chained loop: each call's argument depends on the previous return value.
template <typename Op>
double cycles_per_iter(Op op, uint64_t iters) {
    uint64_t x = 1;
    uint64_t t0 = tsc_begin();
    for (uint64_t i = 0; i < iters; ++i) {
        x = op(x);
        BENCH_VAL(x);
    }
    uint64_t t1 = tsc_end();
    BENCH_VAL(x);
    return double(t1 - t0) / double(iters);   // TSC ticks per iteration
}

// Noise is additive, so the minimum across repetitions is the best estimate.
template <typename Op>
double best_of(Op op, uint64_t iters, int reps = 15) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        double c = cycles_per_iter(op, iters);
        if (c < best) best = c;
    }
    return best;
}

// ---------- Calibration --------------------------------------------------------

// TSC ticks per nanosecond, measured against QueryPerformanceCounter over ~500 ms.
static double ticks_per_ns() {
    LARGE_INTEGER f, a, b;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&a);
    uint64_t t0 = __rdtsc();
    double ns;
    do {
        QueryPerformanceCounter(&b);
        ns = double(b.QuadPart - a.QuadPart) * 1e9 / double(f.QuadPart);
    } while (ns < 5e8);
    uint64_t t1 = __rdtsc();
    return double(t1 - t0) / ns;
}

// TSC ticks per imul. A dependent imul chain is latency-bound at 3 core cycles
// per iteration on Intel and AMD x86-64, which gives the core-clock conversion.
static double ticks_per_imul() {
    constexpr uint64_t N = 50'000'000;
    uint64_t x = 3;
    uint64_t t0 = tsc_begin();
    for (uint64_t i = 0; i < N; ++i)
        asm volatile("imul %0, %0" : "+r"(x));
    uint64_t t1 = tsc_end();
    BENCH_VAL(x);
    return double(t1 - t0) / double(N);
}

// ---------- Main ---------------------------------------------------------------

int main() {
    constexpr uint64_t N = 10'000'000;

    const double tpn = ticks_per_ns();
    const double cyc_per_tick = 3.0 / ticks_per_imul();   // core cycles per TSC tick
    std::printf("TSC %.3f GHz, core %.3f GHz (imul-derived)\n",
                tpn, tpn * cyc_per_tick);

    // Function pointer: opaque, so the compiler emits an indirect call.
    using fn_t = uint64_t (*)(uint64_t);
    fn_t fp = free_leaf;
    BENCH_VAL(fp);

    // std::function: opaque object, so the invoke path cannot be resolved statically.
    std::function<uint64_t(uint64_t)> sf = free_leaf;
    BENCH_MEM(sf);

    // Volatile read hides the dynamic type from the optimizer: no devirtualization.
    Derived d;
    Base* volatile vbp = &d;
    Base* bp = vbp;

    const double base = best_of([](uint64_t v) { return v; }, N);
    std::printf("loop baseline: %.3f ticks/iter = %.3f cycles/iter\n",
                base, base * cyc_per_tick);

    auto report = [&](const char* name, double raw) {
        double net = raw - base;
        std::printf("%-22s %8.3f ticks  %7.3f cycles\n", name, net, net * cyc_per_tick);
    };

    report("direct free fn",    best_of([](uint64_t v) { return free_leaf(v); }, N));
    report("static member",     best_of([](uint64_t v) { return S::sfn(v); }, N));
    report("stack frame only",  best_of([](uint64_t v) { return frame_only(v); }, N));
    report("callee-saved regs", best_of([](uint64_t v) { return callee_saved(v); }, N));
    report("function pointer",  best_of([fp](uint64_t v) { return fp(v); }, N));
    report("std::function",     best_of([&sf](uint64_t v) { return sf(v); }, N));
    report("virtual (mono)",    best_of([bp](uint64_t v) { return bp->f(v); }, N));
    return 0;
}