#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

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

int main(int argc, char const *argv[]) {

    const int LIST_SIZE = 1024;
    int *A = (int *)malloc(sizeof(int) * LIST_SIZE);
    int *B = (int *)malloc(sizeof(int) * LIST_SIZE);

    srand(time(NULL));
    for (int i = 0; i < LIST_SIZE; i++) {
        A[i] = rand() % 100;
    }

    size_t source_size;
    char *source = load_kernel(&source_size);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_int ret;

    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);

    clEnqueueWriteBuffer(queue, a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), A, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &source,
            (const size_t *) &source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log = (char *) malloc(MAX_SOURCE_SIZE);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		printf("%s\n", log);
	}

    cl_kernel kernel = clCreateKernel(program, "k_function", &ret);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    clSetKernelArg(kernel, 2, sizeof(int), &LIST_SIZE);

    size_t global_item_size = LIST_SIZE;
    size_t local_item_size = 64;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size,
            &local_item_size, 0, NULL, NULL);

    clFinish(queue);

    clEnqueueReadBuffer(queue, b_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

    for (int i = 0; i < LIST_SIZE; i++) {
        printf("%d %d\n", A[i], B[i]);
    }

    clFlush(queue);
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    return 0;
}
