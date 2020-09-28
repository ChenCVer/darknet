#include "blas.h"
#include "utils.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int out_w, int out_h, int out_c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int in_c = out_c/(stride*stride);

    // printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
    // printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < out_c; ++k){          // 遍历输出特征图通道数
            for(j = 0; j < out_h; ++j){      // 遍历输出特征图高度
                for(i = 0; i < out_w; ++i){  // 遍历输出特征图宽度, 第b张图片对应的特征图中第k个通道中第j行的第i列元素
                    // in_index = i + j*out_w + k*out_w*out_h + b*out_w*out_h*out_c
                    int in_index  = i + out_w*(j + out_h*(k + out_c*b));
                    // TODO: 疑问, 当k=1,i=0,j=0时, c2=1,w2=0,h2=0, 输出特征图第2特征层对应输入特征图第2个特征层,与图示不符合.
                    int c2 = k % in_c;
                    int offset = k / in_c;                 // 通道偏移
                    int w2 = i*stride + offset % stride;   // 输入特征图宽度位置
                    int h2 = j*stride + offset / stride;   // 输入特征图高度位置
                    // out_index = out_w*out_h*s*s*in_c*b + out_w*out_h*s*s*c2 + out_w*s*h2 + w2
                    int out_index = w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
                    if(forward)
                        out[out_index] = x[in_index];
                    else
                        out[in_index] = x[out_index];  // used by default for forward (i.e. forward = 0)
                }
            }
        }
    }
}


/**
 * @param   x: region层输出结果
 * @param   size: w*h
 * @param   layers: anchor_nums * (coord(4) + num_classes(20)+1)  -->vocdata
 * @param   batch:
 * @param   forward:
 **/
void flatten(float *x, int size, int layers, int batch, int forward)
{   // size: 网格大小, 比如19*19, layers:每个网格中的anchor数x每个anchor需要的预测的数值(coords+class+conf).
    float* swap = (float*)xcalloc(size * layers * batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){    // anchor_nums x (coords+num_class+conf);
            for(i = 0; i < size; ++i){  // 网格大小: 19 x 19;
                // 这里需要理解一下: i1 = b*layers*size + c*size + i
                // b*layers*size表示b-1张图所有的特征, c*size表示已经跨过了c-1张特征图.
                // i表示正在处理的第i个网格.
                // 这里需要理解一下: i2 = b*layers*size + i*layers + c
                // i*layers, 每个网格都占据layers层, 迭代至ixlayers表示已经遍历i个网格的所有层,
                // c表示当前正在处理第c层的特征
                // 综上分析可以, 经过flatten()函数操作之后, l.outputs中的内变为如下形式:
                // size = w*h = 2x2, num_anchors=3, num_class=2;
                // [||-xywhcC1C2-xywhcC1C2-xywhcC1C2-||
                //  ||-xywhcC1C2-xywhcC1C2-xywhcC1C2-||
                //  ||-xywhcC1C2-xywhcC1C2-xywhcC1C2-||
                //  ||-xywhcC1C2-xywhcC1C2-xywhcC1C2-||]
                int i1 = b*layers*size + c*size + i;    // 第b张图片第c层特征图中的第i个网格
                int i2 = b*layers*size + i*layers + c;  // 第b张图片第i个网格中的第c层特征,
                if (forward)
                    swap[i2] = x[i1];
                else
                    swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

/**
 * @param size: int, shortcut总的输出尺寸大小, l.outputs * l.batch
 * @param src_outputs: int, shortcut层输出尺寸大小: l.outputs
 * @param batch:  int, batch大小
 * @param n: int, shortcut的层数, 一般都为1
 * @param outputs_of_layers: int*, l.input_sizes, shortcut层短接的层的输出大小
 * @param layers_output: float**, l.layers_output, shortcut层短接的层的输出的output数据.
 * @param out: float*, l.output, shortcut层的输出数据
 * @param in:  float*, state.input, shortcut自有的输入数据, 这个数据来自于shortcut层的上一层,比如
 *             shortcut层的编号是4,则state.input层就来自于编号为3的层的输出.
 * @param weights: l.weights
 * @param nweights: l.nweights
 * @param weights_normalization:
 */
void shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers,
                             float **layers_output, float *out, float *in, float *weights,
                             int nweights, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {  // size是shortcut总的输出尺寸大小

        int src_id = id;
        // src_id相当于获取第l.batch-1中src_outputs的id
        const int src_i = src_id % src_outputs;  // src_outputs为batch=1时, shortcut层的输出数据大小.
        src_id /= src_outputs;                   // src_id应该是在取l.batch中的batch_id
        int src_b = src_id;                      // 猜测src_b也即: src_batch_id

        float sum = 1, max_val = -FLT_MAX;
        int i;
        if (weights && weights_normalization) {
            if (weights_normalization == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            out[id] = in[id] * w; // [0 or c or (c, h ,w)]
        }
        else out[id] = in[id];

        // layers, 遍历每个短接的层, 这里相当于: 对每个短接的层进行遍历, 然后对应位置相加即为shortcut层输出结果.
        for (i = 0; i < n; ++i) {  // 一般情况: add_outputs = src_outputs
            int add_outputs = outputs_of_layers[i];   // 获取shortcut层的每个短接的层的输出大小.
            if (src_i < add_outputs) {                // 一般情况下: src_i < add_outputs
                // src_b = id / src_outputs;
                // src_i = id % src_outputs;
                // add_index表示第src_b个shortcut输出层中的第i个位置.
                int add_index = add_outputs*src_b + src_i;
                int out_index = id;

                float *add = layers_output[i];  // 获取shortcut层短接的层对应的输出数据.

                if (weights) {
                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    out[out_index] += add[add_index] * w; // [0 or c or (c, h ,w)]
                }
                else out[out_index] += add[add_index];  // 对所有shortcut层短接的层进行通道方向上累加.
            }
        }
    }
}

void backward_shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers,
    float **layers_delta, float *delta_out, float *delta_in, float *weights, float *weight_updates, int nweights,
    float *in, float **layers_output, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {  // size是shortcut总的输出尺寸大小
        int src_id = id;
        int src_i = src_id % src_outputs;
        src_id /= src_outputs;  // src_id相当于获取第l.batch-1中src_outputs的id
        int src_b = src_id;     // 猜测src_b也即: src_batch_id

        float grad = 1, sum = 1, max_val = -FLT_MAX;;
        int i;
        if (weights && weights_normalization) {
            if (weights_normalization == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }

            /*
            grad = 0;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float delta_w = delta_in[id] * in[id];
                const float w = weights[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) grad += delta_w * relu(w) / sum;
                else if (weights_normalization == SOFTMAX_NORMALIZATION) grad += delta_w * expf(w - max_val) / sum;
            }
            */
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            delta_out[id] += delta_in[id] * w; // [0 or c or (c, h ,w)]
            weight_updates[src_i / step] += delta_in[id] * in[id] * grad;
        }
        else delta_out[id] += delta_in[id];

        // layers, 遍历每个短接的层, 这里相当于: 对每个短接的层进行遍历, 然后对应位置相加即为shortcut层输出结果.
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];  // 获取shortcut层的每个短接的层的输出大小.
            if (src_i < add_outputs) {               // 一般情况下: src_i < add_outputs
                int add_index = add_outputs*src_b + src_i;  // add_index表示第src_b个shortcut输出层中的第i个位置.
                int out_index = id;

                float *layer_delta = layers_delta[i];  // 获取shortcut层短接的层对应的l.detla误差项.
                if (weights) {
                    float *add = layers_output[i];

                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    layer_delta[add_index] += delta_in[id] * w; // [0 or c or (c, h ,w)]
                    weight_updates[weights_index] += delta_in[id] * add[add_index] * grad;
                }
                else layer_delta[add_index] += delta_in[id];  // 误差项累计
            }
        }
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

/*
** 计算输入数据x的平均值,输出的mean是一个矢量,比如如果x是多张三通道的图片,那么mean的维度就为通道3
** 由于每次训练输入的都是一个batch的图片,因此最终会输出batch张三通道的图片,mean中的第一个元素就是第
** 一个通道上全部batch张输出特征图所有元素的平均值,本函数的用处之一就是batch normalization的第一步了
** x: 包含所有数据,比如l.output,其包含的元素个数为l.batch*l.outputs
** batch: 一个batch中包含的图片张数,即l.batch
** filters: 该层神经网络的滤波器个数,也即该层网络输出图片的通道数(比如对卷积网络来说,就是核的个数了)
** spatial: 该层神经网络每张输出特征图的尺寸,也即等于l.out_w*l.out_h
** mean: 求得的平均值,维度为filters,也即每个滤波器对应有一个均值(每个滤波器会处理所有图片)
** x的内存排布？此处还是结合batchnorm_layer.c中的forward_batch_norm_layer()函数的调用来解释,其中x为l.output,其包含的元素个数为l
** 有l.batch行,每行有l.out_c*l.out_w*l.out_h个元素,每一行又可以分成l.out_c行,l.out_w*l.out_h列,
** 那么l.mean中的每一个元素,是某一个通道上所有batch的输出的平均值
** (比如卷积层,有3个核,那么输出通道有3个,每张输入图片都会输出3张特征图,可以理解每张输出图片是3通道的,
** 若每次输入batch=64张图片,那么将会输出64张3通道的图片,而mean中的每个元素就是某个通道上所有64张图片
** 所有元素的平均值,比如第1个通道上,所有64张图片像素平均值)
*/
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{   // bn求均值的原理: 比如输出维度(B, C, H, W)为(16, 32, 416, 416), 对于每一个通道C_i
    // 求总共B个H*W方向的总和, 然后除以scale=BxHxW.得到mean_c_i.
    // spatial = l.out_h*l.out_w
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){  // filters也即输出输出通道数
        mean[i] = 0;
        for(j = 0; j < batch; ++j){   // batch
            for(k = 0; k < spatial; ++k){  // out_h * out_w
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

/*
** 计算输入x中每个元素的方差
** 本函数的主要用处应该就是batch normalization的第二步了
** x: 包含所有数据,比如l.output,其包含的元素个数为l.batch*l.outputs
** batch: 一个batch中包含的图片张数,即l.batch
** filters: 该层神经网络的滤波器个数,也即是该网络层输出图片的通道数
** spatial: 该层神经网络每张特征图的尺寸,也即等于l.out_w*l.out_h
** mean: 求得的平均值,维度为filters,也即每个滤波器对应有一个均值(每个滤波器会处理所有图片)
*/
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{   // bn求方差的原理: 比如输出维度(B, C, H, W)为(16, 32, 416, 416), 对于每一个通道C_i
    // 求总共B个H*W方向的总和, 然后除以scale=BxHxW.得到var_c_i.
    // spatial = l.out_h*l.out_w
    // 这里计算方差分母要减去1的原因是无偏估计,可以看：https://www.zhihu.com/question/20983193
    // 事实上,在统计学中,往往采用的方差计算公式都会让分母减1,这时因为所有数据的方差是基于均值这个固定点来计算的,
    // 对于有n个数据的样本,在均值固定的情况下,其采样自由度为n-1(只要n-1个数据固定,第n个可以由均值推出)
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;  // index为batch中的对应通道上的位置.
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

/*
** axpy是线性代数中的一种基本操作(仿射变换)完成y= alpha*x + y操作,其中x,y为矢量,alpha为实数系数,
** 请看: https://www.jianshu.com/p/e3f386771c51
** N: X中包含的有效元素个数
** ALPHA: 系数alpha
** X: 参与运算的矢量X
** INCX: 步长(倍数步长),即x中凡是INCX倍数编号的参与运算
** Y: 参与运算的矢量,也相当于是输出
*/
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{   // N=l.out_c, ALPHA=.1, float* X=l.mean, INCX=1, float* Y=l.rolling_mean; INCY=1
    int i;
    for(i = 0; i < N; ++i) 
        Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{   // N=l.out_c; float ALPHA=0.9; float* X=l.rolling_mean; int INCX=1
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

/*
** 初始化X数组所有元素的值为ALPHA
** N: X中包含的有效元素个数
** X: 待初始化的float数组指针
** INCX: 步长(倍数步长)，即X中凡是INCX的倍数编号进行初始化赋值操作
*/
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    if (INCX == 1 && ALPHA == 0) {
        memset(X, 0, N * sizeof(float));
    }
    else {
        for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];  // t也即target, 真值
        float p = pred[i];   // p也即pred, 网络预测值
        error[i] = (t) ? -log(p) : 0;  // 损失值, t为1时取-log(p); t为0时取0.
        // softmax层的梯度计算∂L/∂z, 其中a = softmax(z), L=-∑t*log(a),
        // 参看:https://blog.csdn.net/abc13526222160/article/details/84968161
        // 这里说明一下:
        // a_i = exp(z_i)/∑k_(exp(z_k));
        // C = -[(y1*log(a_1) + y2*log(a_2)+...+yi*log(a_i)+...+yj*log(a_j)+yn*log(a_n))]
        // 由于a_j中也包含这z_i,因此,我们在求解∂L/∂z_i的时候, 应该要考虑j=i和j≠i两种情况.
        delta[i] = t-p;  // softmax层的梯度.
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

/*
** 输入:  input   一组输入图片数据(含义见下面softmax_cpu()注释,下同)
**       n       一组输入数据中含有的元素个数n=l.inputs/l.groups
**       temp    温度参数,关于softmax的温度参数,可以搜索一下softmax with temperature
**       stride  跨度
**       output  这一组输入图片数据对应的输出(也即l.output中与这一组输入对应的某一部分)
** 说明: 本函数实现的就是标准的softmax函数处理,唯一有点变化的就是在做指数运算之前,将每个输入
 *       元素减去了该组输入元素中的最大值,以增加数值稳定性,关于此,
 *       可以参考博客：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/,
**       这篇博客写的不错,博客中还提到了softmax-loss,此处没有实现(此处实现的也即博客中提到的softmax函数,
 *       将softmax-loss分开实现了)。
*/
void softmax(float *input, int n, float temp, float *output, int stride)
{
    int i;
    float sum = 0;
    // 赋初始最大值为float中的最小值-FLT_MAX
    float largest = -FLT_MAX;
    // 寻找输入中的最大值,至于为什么要找出最大值,是为了数值计算上的稳定,
    // 详细请戳: http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    // 这篇博客写的不错,博客在接近尾声的时候,提到了为什么要减去输入中的最大值。
    for(i = 0; i < n; ++i){   // n也即为类别数: num_classes
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    // 上面说了为啥要减去最大值
    for(i = 0; i < n; ++i){
        // 在进行指数运算之间,如上面博客所说,首先减去最大值(当然温度参数也要除), 主要是为了数值计算上的稳定
        // 关于温度参数的设定: https://www.jianshu.com/p/cb93d5e39bca
        float e = exp(input[i*stride]/temp - largest/temp);  // exp((x-x_max)/temp)
        sum += e;
        output[i*stride] = e;
    }
    // 最后一步: 归一化转换为概率(就是softmax函数的原型),最后的输出结果保存在output中
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

/*
** 对输入Input进行softmax处理得到output
** @input:        softmax层所有的输入数据,即是net.input(上一层的输出)
** @n:            一组数据中含有的元素个数n = l.inputs / l.groups
** @batch:        一个batch中所含有的图片张数,net.batch
** @batch_offset  一张输入图片含有的元素个数,即值等于l.inputs
**                (所以叫做batch_offset,目的是要借助该参数在input中整张整张照片移位)
** @groups:       一张输入图片的元素被分成了几组,值为l.groups(这个参数由配置文件指定,如果未指定,则默认为1)
** @group_offset: 等于n/net.groups,组偏移(在每张输入图片元素中整组整组偏移)
** @stride        跨度,这个参数类似于axpy_cpu()函数中的INCX参数,一定注意不同于卷积层中的l.stride,
 *                这个参数是指按照stride间隔从每组输入数据中抽取元素,即会抽取所有索引为stride倍数的
 *                输入元素,而其他的输入元素,实际没有用到；stride=1时,显然,相当于没有这个参数,所有输
 *                入数据都用到了(这个参数在softmax_layer层中,相当于没用,因为在forward_softmax_layer()中,
 *                调用该函数时,stride已经被写死为1,并不能改,不知道还有没有其他地方使用了这个参数)
** @temp:         softmax的温度参数l.temperature,关于softmax的温度参数,可以搜索一下softmax with temperature,
** @output:       经softmax处理之后得到的输出l.output(即概率),与input具有相同的元素个数(见make_softmax_layer()),
 *                其实由此也可知,stride的值必然为1,不然output的元素个数肯定少于input的元素个数(所以对于softmax来说,
 *                感觉设置stride是没有必要的,有点自相矛盾的意思)这里的groups似乎是有歧义的,因为卷积里面有groups即组
 *                卷积参数,但是这里的意思是一张图片被分成了几份,似乎是AlexNet为了显存够用那个思路但我去查看了AlexNet
 *                的cfg,这里groups也是设置为1的。所以个人认为直接将softmax出现的groups忘掉,然后记住只有分组卷积groups
 *                有用即可。
 * @note: 以上注释针对的是softmax_layer,另有不同地方调用本函数的在调用处进行详细注释；上面的注释出现了新的量词单位,
 *        这里厘清一下关系: 输入input中包括batch中所有图片的输入数据,其中一张图片具有inputs个元素,一张图片的元素又
 *        分成了groups组,每组元素个数为n=l.inputs/l.groups.
*/
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups,
                 int group_offset, int stride, float temp, float *output)
{
    int g, b;
    // 遍历batch中的每张图片
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            // 每张图片又按组遍历: 一组一组遍历
            softmax(input + b*batch_offset + g*group_offset, n, temp,
                    output + b*batch_offset + g*group_offset, stride);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{   // 反向传播的时候, float *in即为state.delta, 表示当前层的上一层的误差项, 是需要求的. float *out当前层误差项.
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if (forward)
                        out[out_index] = scale*in[in_index];
                    else
                        in[in_index] += scale*out[out_index];  // 使用+=, 进行累计误差!
                }
            }
        }
    }
}


void constrain_cpu(int size, float ALPHA, float *X)
{
    int i;
    for (i = 0; i < size; ++i) {
        X[i] = fminf(ALPHA, fmaxf(-ALPHA, X[i]));
    }
}

void fix_nan_and_inf_cpu(float *input, size_t size)
{
    int i;
    for (i = 0; i < size; ++i) {
        float val = input[i];
        if (isnan(val) || isinf(val))
            input[i] = 1.0f / i;  // pseudo random value
    }
}

void get_embedding(float *src, int src_w, int src_h, int src_c, int embedding_size, int cur_w, int cur_h, int cur_n, int cur_b, float *dst)
{
    int i;
    for (i = 0; i < embedding_size; ++i) {
        const int src_index = cur_b*(src_c*src_h*src_w) + cur_n*(embedding_size*src_h*src_w) + i*src_h*src_w + cur_h*(src_w) + cur_w;

        const float val = src[src_index];
        dst[i] = val;
        //printf(" val = %f, ", val);
    }
}


// Euclidean_norm
float math_vector_length(float *A, unsigned int feature_size)
{
    float sum = 0;
    int i;
    for (i = 0; i < feature_size; ++i)
    {
        sum += A[i] * A[i];
    }
    float vector_length = sqrtf(sum);
    return vector_length;
}

float cosine_similarity(float *A, float *B, unsigned int feature_size)
{
    float mul = 0.0, d_a = 0.0, d_b = 0.0;

    int i;
    for(i = 0; i < feature_size; ++i)
    {
        mul += A[i] * B[i];
        d_a += A[i] * A[i];
        d_b += B[i] * B[i];
    }
    float similarity;
    float divider = sqrtf(d_a) * sqrtf(d_b);
    if (divider > 0) similarity = mul / divider;
    else similarity = 0;

    return similarity;
}

int get_sim_P_index(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        return -1;   // not found
    }

    return z;   // found
}

int check_sim(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        return 0;   // not found
    }

    return 1;   // found
}

float find_sim(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        printf(" Error: find_sim(): sim isn't found: i = %d, j = %d, z = %d \n", i, j, z);
        getchar();
    }

    return contrast_p[z].sim;
}

float find_P_constrastive(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        printf(" Error: find_P_constrastive(): P isn't found: i = %d, j = %d, z = %d \n", i, j, z);
        getchar();
    }

    return contrast_p[z].P;
}

// num_of_samples = 2 * loaded_images = mini_batch_size
float P_constrastive_f_det(size_t il, int *labels, float **z, unsigned int feature_size, float temperature, contrastive_params *contrast_p, int contrast_p_size)
{
    const float sim = contrast_p[il].sim;
    const size_t i = contrast_p[il].i;
    const size_t j = contrast_p[il].j;

    const float numerator = expf(sim / temperature);

    float denominator = 0;
    int k;
    for (k = 0; k < contrast_p_size; ++k) {
        contrastive_params cp = contrast_p[k];
        //if (k != i && labels[k] != labels[i]) {
        //if (k != i) {
        if (cp.i != i && cp.j == j) {
            //const float sim_den = cp.sim;
            ////const float sim_den = find_sim(k, l, contrast_p, contrast_p_size); // cosine_similarity(z[k], z[l], feature_size);
            //denominator += expf(sim_den / temperature);
            denominator += cp.exp_sim;
        }
    }

    float result = 0.9999;
    if (denominator != 0) result = numerator / denominator;
    if (result > 1) result = 0.9999;
    return result;
}

// num_of_samples = 2 * loaded_images = mini_batch_size
float P_constrastive_f(size_t i, size_t l, int *labels, float **z, unsigned int feature_size, float temperature, contrastive_params *contrast_p, int contrast_p_size)
{
    if (i == l) {
        fprintf(stderr, " Error: in P_constrastive must be i != l, while i = %d, l = %d \n", i, l);
        getchar();
    }

    const float sim = find_sim(i, l, contrast_p, contrast_p_size); // cosine_similarity(z[i], z[l], feature_size);
    const float numerator = expf(sim / temperature);

    float denominator = 0;
    int k;
    for (k = 0; k < contrast_p_size; ++k) {
        contrastive_params cp = contrast_p[k];
        //if (k != i && labels[k] != labels[i]) {
        //if (k != i) {
        if (cp.i != i && cp.j == l) {
            //const float sim_den = cp.sim;
            ////const float sim_den = find_sim(k, l, contrast_p, contrast_p_size); // cosine_similarity(z[k], z[l], feature_size);
            //denominator += expf(sim_den / temperature);
            denominator += cp.exp_sim;
        }
    }

    float result = 0.9999;
    if (denominator != 0) result = numerator / denominator;
    if (result > 1) result = 0.9999;
    return result;
}

void grad_contrastive_loss_positive_f(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *delta, int wh, contrastive_params *contrast_p, int contrast_p_size)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j] && labels[i] >= 0) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d \n",
            N, temperature, vec_len, labels[i]);
        getchar();
        return;
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j] && labels[i] >= 0) {
            //printf(" i = %d, j = %d, num_of_samples = %d, labels[i] = %d, labels[j] = %d \n",
            //    i, j, num_of_samples, labels[i], labels[j]);
            const int sim_P_i = get_sim_P_index(i, j, contrast_p, contrast_p_size);
            if (sim_P_i < 0) continue;
            const float sim = contrast_p[sim_P_i].sim;
            const float P = contrast_p[sim_P_i].P;
            //if (!check_sim(i, j, contrast_p, contrast_p_size)) continue;
            //const float sim = find_sim(i, j, contrast_p, contrast_p_size); //cos_sim[i*num_of_samples + j];        // cosine_similarity(z[i], z[j], feature_size);
            //const float P = find_P_constrastive(i, j, contrast_p, contrast_p_size); //p_constrastive[i*num_of_samples + j];   // P_constrastive(i, j, labels, num_of_samples, z, feature_size, temperature, cos_sim);
                                                                    //const float custom_pos_mult = 1 - sim;


            int m;
            //const float d = mult*(sim * z[i][m] - z[j][m]) * (1 - P); // 1
            for (m = 0; m < feature_size; ++m) {
                //const float d = mult*(sim * z[j][m] - z[j][m]) * (1 - P); // my
                //const float d = mult*(sim * z[i][m] + sim * z[j][m] - z[j][m]) *(1 - P); // 1+2
                const float d = mult*(sim * z[i][m] - z[j][m]) *(1 - P); // 1 (70%)
                //const float d = mult*(sim * z[j][m] - z[j][m]) * (1 - P); // 2
                // printf(" pos: z[j][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[j][m], z[i][m], d, sim);
                const int out_i = m * wh;
                delta[out_i] -= d;
            }
        }
    }
}

void grad_contrastive_loss_negative_f(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *delta, int wh, contrastive_params *contrast_p, int contrast_p_size)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j] && labels[i] >= 0) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d \n",
            N, temperature, vec_len, labels[i]);
        getchar();
        return;
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j] && labels[i] >= 0) {

            size_t k;
            for (k = 0; k < num_of_samples; ++k) {
                //if (k != i && k != j && labels[k] != labels[i]) {
                if (k != i && k != j && labels[k] >= 0) {
                    const int sim_P_i = get_sim_P_index(i, k, contrast_p, contrast_p_size);
                    if (sim_P_i < 0) continue;
                    const float sim = contrast_p[sim_P_i].sim;
                    const float P = contrast_p[sim_P_i].P;
                    //if (!check_sim(i, k, contrast_p, contrast_p_size)) continue;
                    //const float sim = find_sim(i, k, contrast_p, contrast_p_size); //cos_sim[i*num_of_samples + k];        // cosine_similarity(z[i], z[k], feature_size);
                    //const float P = find_P_constrastive(i, k, contrast_p, contrast_p_size); //p_constrastive[i*num_of_samples + k];   // P_constrastive(i, k, labels, num_of_samples, z, feature_size, temperature, cos_sim);
                                                                            //const float custom_pos_mult = 1 + sim;

                    int m;
                    //const float d = mult*(z[k][m] + sim * z[i][m]) * P;   // my1
                    for (m = 0; m < feature_size; ++m) {
                        //const float d = mult*(z[k][m] + sim * z[i][m]) * P;   // 1 (70%)
                        //const float d = mult*(z[k][m] - sim * z[k][m] - sim * z[i][m]) * P;   // 1+2
                        const float d = mult*(z[k][m] - sim * z[i][m]) * P;   // 1 (70%)
                        //const float d = mult*(z[k][m] - sim * z[k][m]) * P; // 2
                        //printf(" neg: z[k][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[k][m], z[i][m], d, sim);
                        const int out_i = m * wh;
                        delta[out_i] -= d;
                    }
                }
            }
        }
    }
}



// num_of_samples = 2 * loaded_images = mini_batch_size
float P_constrastive(size_t i, size_t l, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *exp_cos_sim)
{
    if (i == l) {
        fprintf(stderr, " Error: in P_constrastive must be i != l, while i = %d, l = %d \n", i, l);
        getchar();
    }

    //const float sim = cos_sim[i*num_of_samples + l]; // cosine_similarity(z[i], z[l], feature_size);
    //const float numerator = expf(sim / temperature);
    const float numerator = exp_cos_sim[i*num_of_samples + l];

    float denominator = 0;
    int k;
    for (k = 0; k < num_of_samples; ++k) {
        //if (k != i && labels[k] != labels[i]) {
        if (k != i) {
            //const float sim_den = cos_sim[k*num_of_samples + l]; // cosine_similarity(z[k], z[l], feature_size);
            //denominator += expf(sim_den / temperature);
            denominator += exp_cos_sim[k*num_of_samples + l];
        }
    }

    float result = numerator / denominator;
    return result;
}

// i - id of the current sample in mini_batch
// labels[num_of_samples] - array with class_id for each sample in the current mini_batch
// z[feature_size][num_of_samples] - array of arrays with contrastive features (output of conv-layer, f.e. 128 floats for each sample)
// delta[feature_size] - array with deltas for backpropagation
// temperature - scalar temperature param (temperature > 0), f.e. temperature = 0.07: Supervised Contrastive Learning
void grad_contrastive_loss_positive(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *p_constrastive, float *delta, int wh)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j]) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f \n", N, temperature, vec_len);
        getchar();
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j]) {
            //printf(" i = %d, j = %d, num_of_samples = %d, labels[i] = %d, labels[j] = %d \n",
            //    i, j, num_of_samples, labels[i], labels[j]);
            const float sim = cos_sim[i*num_of_samples + j];        // cosine_similarity(z[i], z[j], feature_size);
            const float P = p_constrastive[i*num_of_samples + j];   // P_constrastive(i, j, labels, num_of_samples, z, feature_size, temperature, cos_sim);
            //const float custom_pos_mult = 1 - sim;

            int m;
            for (m = 0; m < feature_size; ++m) {
                const float d = mult*(sim * z[i][m] - z[j][m]) * (1 - P); // good
                //const float d = mult*(sim * z[j][m] - z[j][m]) * (1 - P); // bad
               // printf(" pos: z[j][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[j][m], z[i][m], d, sim);
                const int out_i = m * wh;
                delta[out_i] -= d;
            }
        }
    }
}

// i - id of the current sample in mini_batch
// labels[num_of_samples] - array with class_id for each sample in the current mini_batch
// z[feature_size][num_of_samples] - array of arrays with contrastive features (output of conv-layer, f.e. 128 floats for each sample)
// delta[feature_size] - array with deltas for backpropagation
// temperature - scalar temperature param (temperature > 0), f.e. temperature = 0.07: Supervised Contrastive Learning
void grad_contrastive_loss_negative(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *p_constrastive, float *delta, int wh)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j]) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f \n", N, temperature, vec_len);
        getchar();
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j]) {

            size_t k;
            for (k = 0; k < num_of_samples; ++k) {
                //if (k != i && k != j && labels[k] != labels[i]) {
                if (k != i && k != j && labels[k] >= 0) {
                    const float sim = cos_sim[i*num_of_samples + k];        // cosine_similarity(z[i], z[k], feature_size);
                    const float P = p_constrastive[i*num_of_samples + k];   // P_constrastive(i, k, labels, num_of_samples, z, feature_size, temperature, cos_sim);
                    //const float custom_pos_mult = 1 + sim;

                    int m;
                    for (m = 0; m < feature_size; ++m) {
                        const float d = mult*(z[k][m] - sim * z[i][m]) * P;   // good
                        //const float d = mult*(z[k][m] - sim * z[k][m]) * P; // bad
                        //printf(" neg: z[k][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[k][m], z[i][m], d, sim);
                        const int out_i = m * wh;
                        delta[out_i] -= d;
                    }
                }
            }
        }
    }
}