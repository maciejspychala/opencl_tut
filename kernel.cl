#define arr(x, y) graph[x * graph_size + y]
#define min(a, b) a > b ? b : a
__kernel void k_function(__global int *graph, const int graph_size, const int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i > graph_size - 1) return;
    if(j > graph_size - 1) return;
    arr(i, j) = min(arr(i, k) + arr(k, j), arr(i, j));
}
