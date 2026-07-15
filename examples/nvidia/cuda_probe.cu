// CUDA phased workload (GPU-heavy then CPU-heavy), for validating the NVIDIA
// kernel tracer + CPU/GPU timeline on the same CUPTI clock.
#include <cstdio>
#include <chrono>
#include <thread>
#include <vector>

__global__ void busy(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < 128; k++) x = x * 1.001f + 0.5f;
        a[i] = x;
    }
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void cpu_burn(double seconds) {
    double t0 = now_s();
    volatile double x = 0;
    while (now_s() - t0 < seconds) {
        for (int i = 0; i < 200000; i++) x += i * 1.0001;
    }
}

int main() {
    int n = 1 << 22;
    size_t sz = n * sizeof(float);
    float* d = nullptr;
    cudaMalloc(&d, sz);
    cudaMemset(d, 0, sz);
    int threads = 256, blocks = (n + threads - 1) / threads;
    unsigned ncpu = std::thread::hardware_concurrency();
    if (ncpu < 1) ncpu = 8;

    for (int cyc = 0; cyc < 3; cyc++) {
        double g0 = now_s();
        while (now_s() - g0 < 0.6) {
            busy<<<blocks, threads>>>(d, n);
            cudaDeviceSynchronize();
        }
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < ncpu; t++) pool.emplace_back(cpu_burn, 0.6);
        for (auto& th : pool) th.join();
    }
    cudaFree(d);
    printf("done: 3 cycles GPU-heavy + CPU-heavy\n");
    return 0;
}
