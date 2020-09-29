#include "route_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int groups, int group_id)
{   // route层是将某些层的输出拼接在一起, 然后输出. 比如YOLO V2的第25层为route层, 他连接的是第16层, 则就是将第16层的输出结果
    // 复制过来, 作为输出.
    fprintf(stderr,"route ");
    route_layer l = { (LAYER_TYPE)0 };
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;                        // 这里指的是有多少个层参与拼接
    l.input_layers = input_layers;  // 指定要参与拼接的层
    l.input_sizes = input_sizes;    // 指定输入数据大小
    l.groups = groups;
    l.group_id = group_id;
    int i;
    int outputs = 0;  // 输出元素个数.
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);  // input_layers[i]即为网络层编号
        outputs += input_sizes[i];
    }
    outputs = outputs / groups;
    l.outputs = outputs;
    l.inputs = outputs;  //
    // fprintf(stderr, " inputs = %d \t outputs = %d, groups = %d, group_id = %d \n", l.inputs, l.outputs, l.groups, l.group_id);
    l.delta = (float*)xcalloc(outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(outputs * batch, sizeof(float));

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    #ifdef GPU
    l.forward_gpu = forward_route_layer_gpu;
    l.backward_gpu = backward_route_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("Error: Different size of input layers: %d x %d, %d x %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
            exit(EXIT_FAILURE);
        }
    }
    l->out_c = l->out_c / l->groups;
    l->outputs = l->outputs / l->groups;
    l->inputs = l->outputs;
    l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif

}

void forward_route_layer(const route_layer l, network_state state)
{   // yolo v2: 这一层实在合并27和24层. 以及route 16
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];  // 获取指定层的id
        // 获取指定层的输出, 假设是YOLO v2, route 16, 则这里是获取第16层的输出, 作为route层的输入
        float *input = state.net.layers[index].output;
        // 输入尺寸大小
        int input_size = l.input_sizes[i];  // 指定层的输入
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            // l.output + offset + j*l.outputs = input + j*input_size + part_input_size*l.group_id
            copy_cpu(part_input_size, input + j*input_size + part_input_size*l.group_id, 1,
                     l.output + offset + j*l.outputs, 1);
        }
        //offset += input_size;
        offset += part_input_size;  // 计算位置偏移量
    }
}

void backward_route_layer(const route_layer l, network_state state)
{   // 以YOLO v2中第28层为例, route层, 它是将27和24层的输出特征图进行concat操作.
    // 由反向传播我们首先是求出了第28层的误差项, 然后route相当于是等值传递给27和24层.
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        // 第27, 28层的误差项地址, 这个误差项是只有δ_l*W部分的, 还没乘以f'(net)这块
        float *delta = state.net.layers[index].delta;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            // delta = delta+ALPHA*(l.delta+offset+j*l.outputs);
            // TODO: 这里使用“+=”操作, 也是计算累计误差项.
            axpy_cpu(part_input_size, 1, l.delta + offset + j*l.outputs, 1,
                     delta + j*input_size + part_input_size*l.group_id, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = state.net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            //copy_ongpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
            //simple_copy_ongpu(input_size, input + j*input_size, l.output_gpu + offset + j*l.outputs);
            simple_copy_ongpu(part_input_size, input + j*input_size + part_input_size*l.group_id, l.output_gpu + offset + j*l.outputs);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}

void backward_route_layer_gpu(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = state.net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            //axpy_ongpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
            axpy_ongpu(part_input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size + part_input_size*l.group_id, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}
#endif
