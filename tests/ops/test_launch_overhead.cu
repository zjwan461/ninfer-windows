// What a kernel launch costs on this device, measured against the same work in a graph.
//
// The decode path issues roughly a thousand small kernels per token, so the launch cost is a
// first-class number here rather than a detail: on a shared or partitioned device it can exceed the
// arithmetic. This measures it directly - the same trivial kernel N times, once launched one by one
// and once captured into a graph and replayed.

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <cstdio>


namespace {

__global__ void trivial(int* value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { *value += 1; }
}

double seconds_for(const std::function<void()>& body, int repeats) {
    cudaDeviceSynchronize();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) { body(); }
    cudaDeviceSynchronize();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

} // namespace

int main() {
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;
    }
    cudaMemset(device, 0, sizeof(int));

    constexpr int kNodes = 1000;
    constexpr int kRepeats = 20;

    const double launched = seconds_for([&] {
        for (int i = 0; i < kNodes; ++i) { trivial<<<1, 32>>>(device); }
    }, kRepeats);

    // Capture the same sequence once, then replay it kRepeats times.
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaGraph_t graph = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < kNodes; ++i) { trivial<<<1, 32, 0, stream>>>(device); }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t executable = nullptr;
    cudaGraphInstantiateWithFlags(&executable, graph, 0);
    const double replayed = seconds_for([&] { cudaGraphLaunch(executable, stream); }, kRepeats);

    const double per_launch = launched / (kRepeats * kNodes) * 1e6;
    const double per_node = replayed / (kRepeats * kNodes) * 1e6;
    std::printf("launched  %.3f s for %d kernels -> %.2f us each\n", launched, kRepeats * kNodes,
                per_launch);
    std::printf("graph     %.3f s for %d nodes   -> %.2f us each\n", replayed, kRepeats * kNodes,
                per_node);
    std::printf("%s: a launch costs %.1fx a graph node\n",
                per_node > 0 && per_launch / per_node > 3 ? "LAUNCH-BOUND" : "not launch-bound",
                per_node > 0 ? per_launch / per_node : 0.0);

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(device);
    return 0;
}
