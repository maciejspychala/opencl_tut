__kernel void k_function( __global const int *A, __global int *B, const int N) {
    int i = get_global_id(0);
	if(i > N - 1) return;
	int left = i > 0 ? A[i - 1] > A[i] : 1;
	int right = i < N - 1 ? A[i + 1] > A[i] : 1;
    B[i] = left && right;
}
