__kernel void k_function( __global const int *graph, __global int *output, const int graph_size) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x > graph_size - 1) return;
    if(y > graph_size - 1) return;
    output[x * graph_size + y] = graph[x * graph_size + y] + 1;
}
