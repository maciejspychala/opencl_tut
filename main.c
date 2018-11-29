#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#define MAX_SOURCE_SIZE (0x100000)

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

const int GRAPH_SIZE = 1000;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;
cl_mem graph_dev;
cl_context context;

#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)
#define TIMER_START() gettimeofday(&tv1, NULL)
#define TIMER_STOP()\
    gettimeofday(&tv2, NULL);\
    timersub(&tv2, &tv1, &tv);\
    time_delta = (float)tv.tv_sec + tv.tv_usec / 1000000.0;\
    fprintf(stderr, "%f secs\n", time_delta)

#define INF 0x1fffffff

char* load_kernel(size_t *source_size) {
    FILE *fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char *source = (char*) malloc(MAX_SOURCE_SIZE);
    *source_size = fread(source, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    return source;
}

void generate_random_graph(int *output, int graph_size) {
    srand(0xdadadada);
    for (int i = 0; i < graph_size; i++) {
        for (int j = 0; j < graph_size; j++) {
            int r = i == j ? 0 : rand() % 40;
            r = r > 20 ? INF : r;
            D(i, j) = r;
        }
    }
}

void floyd_warshall_gpu(const int *graph, int graph_size, int *output) {
    size_t source_size;
    char *source = load_kernel(&source_size);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_int ret;

    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    queue = clCreateCommandQueue(context, device_id, 0, &ret);

    graph_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, GRAPH_SIZE * GRAPH_SIZE * sizeof(int), NULL, &ret);
    ret = clEnqueueWriteBuffer(queue, graph_dev, CL_TRUE, 0, graph_size * graph_size * sizeof(int), graph, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, (const char**) &source, (const size_t *) &source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log = (char *) malloc(MAX_SOURCE_SIZE);
        ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        printf("log: %s\n", log);
        printf("source: %s\n", source);
    }

    kernel = clCreateKernel(program, "k_function", &ret);
    for (int k = 0; k < GRAPH_SIZE; k++) {
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&graph_dev);
        ret = clSetKernelArg(kernel, 1, sizeof(int), &GRAPH_SIZE);
        ret = clSetKernelArg(kernel, 2, sizeof(int), &k);

        size_t global_item_size[] = { GRAPH_SIZE, GRAPH_SIZE };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);

        clFinish(queue);
    }
    clEnqueueReadBuffer(queue, graph_dev, CL_TRUE, 0, GRAPH_SIZE * GRAPH_SIZE * sizeof(int), output, 0, NULL, NULL);
}

void floyd_warshall_cpu(const int *graph, int graph_size, int *output) {
    int i, j, k;

    memcpy(output, graph, sizeof(int) * graph_size * graph_size);

    for (k = 0; k < graph_size; k++) {
        for (i = 0; i < graph_size; i++) {
            for (j = 0; j < graph_size; j++) {
                if (D(i, k) + D(k, j) < D(i, j)) {
                    D(i, j) = D(i, k) + D(k, j);
                }
            }
        }
    }
}

int main(int argc, char const *argv[]) {
    struct timeval tv1, tv2, tv;
    float time_delta;

    int *graph = calloc(sizeof(int), GRAPH_SIZE * GRAPH_SIZE);
    int *output_cpu = calloc(sizeof(int), GRAPH_SIZE * GRAPH_SIZE);
    int *output_gpu = calloc(sizeof(int), GRAPH_SIZE * GRAPH_SIZE);
    assert(graph);
    assert(output_cpu);
    assert(output_gpu);

    generate_random_graph(graph, GRAPH_SIZE);

    fprintf(stderr, "running on cpu...\n");
    TIMER_START();
    floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
    TIMER_STOP();

    fprintf(stderr, "running on gpu...\n");
    TIMER_START();
    floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);
    TIMER_STOP();

    if (memcmp(output_cpu, output_gpu, GRAPH_SIZE * GRAPH_SIZE) != 0) {
        fprintf(stderr, "FAIL!\n");
    } else {
        printf("correct\n");
    }

    clFlush(queue);
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(graph_dev);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(graph);
    free(output_cpu);
    return 0;
}
