#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define DOABS 1

// 构建YOLOv2 region_layer层
// batch 一个batch中包含的图片数
// w 输入特征图的宽度
// h 输入特征图的高度
// n 一个cell预测多少个bbox
// classes 网络需要识别的物体类别数
// coord 一个bbox包含的[x,y,w,h]
region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes)
{
    region_layer l = { (LAYER_TYPE)0 };
    l.type = REGION;

    l.n = n;              // anchors_nums,
    l.batch = batch;      // mini_batch大小
    l.h = h;              // 输出特征图高度
    l.w = w;              // 输出特征图高度
    l.classes = classes;  // 预测类别数
    l.coords = coords;    // 一个边界框的参数(x,y,w,h)
    l.cost = (float*)xcalloc(1, sizeof(float));        // 损失值
    l.biases = (float*)xcalloc(n * 2, sizeof(float));  // 用来存储anchor的宽高信息.
    l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
    // 一张图片经过backbone网络之后, 输出的特征量大小: out_w*out_h*num_anchors*(num_cls + num_coords + conf)
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;     // 对于region_layer, 输入和输出的元素个数相等
    l.max_boxes = max_boxes;  // 一张图片中允许的最大gt数目
    l.truth_size = 4 + 2;     // 一个gt框包含的参数信息: x,y,w,h,cls_id, 这里有6个, 不知道第6个参数是什么
    l.truths = max_boxes*l.truth_size;
    l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));  // 误差项
    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){  // 初始化
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(time(0));

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
#ifdef GPU
    int old_w = l->w;
    int old_h = l->h;
#endif
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
    l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef GPU
    //if (old_w < w || old_h < h)
    {
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
        l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#endif
}

//获取某个矩形框的4个定位信息,即根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h
//x      region_layer的输出,即l.output,包含所有batch预测得到的矩形框信息
//biases 表示Anchor框的长和宽
//index  矩形框的首地址(索引,矩形框中存储的首个参数x在l.output中的索引)
//i 第几行(region_layer维度为l.out_w*l.out_c)
//j 第几列
//w 特征图的宽度
//h 特征图的高度
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;  // b.x = (c_x + sigmoid(t_x))
    b.y = (j + logistic_activate(x[index + 1])) / h;  // b.y = (c_y + sigmoid(t_y))
    b.w = exp(x[index + 2]) * biases[2*n];            // b.w = anchor_w * exp(t_w)
    b.h = exp(x[index + 3]) * biases[2*n+1];          // b.h = anchor_h * exp(t_h)
    if(DOABS){   // TODO: DOABS是做什么?
        b.w = exp(x[index + 2]) * biases[2*n]   / w;
        b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    }
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    // 获得第j*w+i个cell第n个bbox在当前特征图上位置和宽高
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    // 计算pred bbox 与 GT bbox的IOU【前12800GT boox为当前cell第n个bbox的Anchor】
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w / biases[2*n]);
    float th = log(truth.h / biases[2*n + 1]);
    if(DOABS){
        tw = log(truth.w*w / biases[2*n]);
        th = log(truth.h*h / biases[2*n + 1]);
    }
    // 这里记录的是坐标损失([truth_r - b_ij_r]^2, 其中truth_r表示gt与anchor之间的变换量(tx,ty,tw,th),
    // b_ij_r也即x[index + 0]~x[index+3]表示网络预测值, 该值表示anchor与预测框之间的变换量)对应的梯度信息:
    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
    return iou;
}

void delta_region_class(float *output, float *delta, int index, int class_id, int classes, tree *hier, float scale, float *avg_cat, int focal_loss)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class_id >= 0){
            pred *= output[index + class_id];
            int g = hier->group[class_id];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + offset + i] = scale * (0 - output[index + offset + i]);
            }
            delta[index + class_id] = scale * (1 - output[index + class_id]);

            class_id = hier->parent[class_id];
        }
        *avg_cat += pred;
    } else {
        // Focal loss
        if (focal_loss) {
            // Focal Loss
            float alpha = 0.5;    // 0.25 or 0.5
            //float gamma = 2;    // hardcoded in many places of the grad-formula

            int ti = index + class_id;
            float pt = output[ti] + 0.000000000000001F;
            // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
            float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
            //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);

                delta[index + n] *= alpha*grad;

                if (n == class_id) *avg_cat += output[index + n];
            }
        }
        else {
            // default
            for (n = 0; n < classes; ++n) {
                // delta[index + n]记录着类别损失对应的梯度信息.
                // 基于softmax计算交叉熵损失,其梯度表示为: t - p;
                // 其中target和pred均为one-hot编码形式.output[index + n]即为该pred_bbox预测为某一类物体的概率值.
                // https://blog.csdn.net/jasonleesjtu/article/details/89426465
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);
                if (n == class_id) *avg_cat += output[index + n];
            }
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);

/** 本函数是yolo系列的损失函数, 其中需要理解的主要是flatten()函数是怎样改变l.outputs的内存分布,
 *  从forward_region_layer()多处for循环可以看出, 比如:
 *  for(i=0; i<l.h*l.w*l.n; i++)
 *      int index = size * i + b*l.outputs;
 *  总共有l.h*l.w*l.n预测框, 每个预测框占据size=l.coords + l.classes + 1空间大小.并且结合
 *  flatten()函数可以推出,l.outputs的内存分布形式.
 *
 * **/
void forward_region_layer(const region_layer l, network_state state)
{
    int i,j,b,t,n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    #ifndef GPU
    // 网格大小, l.n表示anchor数, size表示每个anchor中需要预测的数值个数(l.coords[4]+l.classes[20]+conf[1]);
    flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    #endif
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){      // 遍历每一个anchor
            // 每个anchor在outputs中占据size大小的空间位置, 这里index是查找到每个anchor占据内存空间outputs的偏移量
            int index = size*i + b*l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);  // 获取conf(pred_bbox与gt的IOU), 并做sigmoid()
        }
    }


#ifndef GPU
    if (l.softmax_tree){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1,
                             l.softmax_tree, l.output + index + 5);
            }
        }
    } else if (l.softmax){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){      // 遍历每一个anchor, anchor中的所有类别值过softmax函数得到对应的概率值
                int index = size*i + b*l.outputs;  // 解释同上
                // 这里是对预测类别得分进行softmax()操作, 得到对应的概率值
                softmax(l.output + index + 5, l.classes,
                        1, l.output + index + 5, 1);
            }
        }
    }
#endif
    if(!state.train) return;
    // 核心: 每次forward()之后都会对region层的梯度进行清零处理.
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));  // l.delta中梯度清零
    float avg_iou = 0;     // 所有gt与最佳anchor的之间都有iou, 然后avg += iou, 然后除以count
    float recall = 0;      // 所有gt与对应的l.n个预测框中某一个的iou>0.5,则recall+1, 然后除以count
    float avg_cat = 0;
    float avg_obj = 0;     // 所有与gt对应的最佳anchor对应的pred_bbox的conf值的累加, 然后除以count
    float avg_anyobj = 0;  // 所有pred_bbox输出的conf的累加值, 然后除以count, 无实质用处
    int count = 0;         // 一张图片中有效的gt总数
    int class_count = 0;
    *(l.cost) = 0;         // 损失值也必须清零处理.
    for (b = 0; b < l.batch; ++b) {

        if(l.softmax_tree){
            int onlyclass_id = 0;
            for(t = 0; t < l.max_boxes; ++t){
                box truth = float_to_box(state.truth + t*l.truth_size + b*l.truths);
                if(!truth.x) break; // continue;
                int class_id = state.truth[t*l.truth_size + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int index = size*n + b*l.outputs + 5;
                        float scale =  l.output[index-1];
                        float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class_id);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int index = size*maxi + b*l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class_id, l.classes, l.softmax_tree,
                                       l.class_scale, &avg_cat, l.focal_loss);
                    ++class_count;
                    onlyclass_id = 1;
                    break;
                }
            }
            if(onlyclass_id) continue;
        }

        // 遍历每一个网格中的每一个anchor
        for (j = 0; j < l.h; ++j) {          // 遍历网格高
            for (i = 0; i < l.w; ++i) {      // 遍历网格宽
                for (n = 0; n < l.n; ++n) {  // 遍历每一个anchor
                    // 每一个anchor都在l.outputs上占据size大小的空间, index找到对应anchor的空间偏移位置
                    int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                    // 得到该anchor对应的预测框
                    box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                    float best_iou = 0;
                    int best_class_id = -1;
                    // 每一个预测框与该图片中所有的gt计算iou
                    for(t = 0; t < l.max_boxes; ++t){
                        // 获取gt信息
                        box truth = float_to_box(state.truth + t*l.truth_size + b*l.truths);
                        // 得到gt对应类别id
                        int class_id = state.truth[t * l.truth_size + b*l.truths + 4];
                        if (class_id >= l.classes) continue;
                        // 遍历完所有的gt,及时退出for循环
                        if(!truth.x) break; // continue;
                        float iou = box_iou(pred, truth);  // 计算pred与gt的iou
                        if (iou > best_iou) {
                            best_class_id = state.truth[t*l.truth_size + b*l.truths + 4];  // 标记该pred_bbox对应的类别
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += l.output[index + 4];
                    // 这里相当于初始化l.delta[index+4]这个位置, 是负样本对应的梯度, 也即anchor对应的pred_bbox与所有的gt的IOU
                    // 都小于预设值: 1_maxIOU<thresh × λ_noobj x (0 - b_ijk_o)^2, 其中b_ijk_o即为预测框与gt之间confidence,
                    // 这里是对应负样本的iou损失, 因为conf采用的是sigmoid函数得到, 因此l.delta记录梯度时乘以sigmoid函数的梯度
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) *
                                         logistic_gradient(l.output[index + 4]));
                    if(l.classfix == -1)
                        l.delta[index + 4] = l.noobject_scale * ((best_iou - l.output[index + 4]) *
                                         logistic_gradient(l.output[index + 4]));
                    else{
                        if (best_iou > l.thresh) {
                            // 这里是忽略样本, 不是正样本
                            l.delta[index + 4] = 0;
                            if(l.classfix > 0){
                                delta_region_class(l.output, l.delta, index + 5, best_class_id,l.classes,
                                                   l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + 4] : 1),
                                                   &avg_cat, l.focal_loss);
                                ++class_count;
                            }
                        }
                    }
                    // 这里就是当看的图片数量小于12800时, 这里是预测框与anchor的误差,而不是与gt的误差,
                    // 是为了在训练前期使预测框快速学习到先验框的形状.在很多新复现的论文中没有加入这个loss。
                    if(*(state.net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n];
                        truth.h = l.biases[2*n+1];
                        if(DOABS){
                            truth.w = l.biases[2*n]/l.w;
                            truth.h = l.biases[2*n+1]/l.h;
                        }
                        delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                    }
                }
            }
        }

        // 遍历一张图片中的所有gt
        for(t = 0; t < l.max_boxes; ++t){
            // 获取gt信息
            box truth = float_to_box(state.truth + t*l.truth_size + b*l.truths);
            // 获取gt对应的类别信息class_id.
            int class_id = state.truth[t * l.truth_size + b*l.truths + 4];
            if (class_id >= l.classes) {
                printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. "
                       "In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes-1);
                getchar();
                continue; // if label contains class_id more than number of classes in the cfg-file
            }
            // 遍历完所有的gt, 即时退出
            if(!truth.x) break; // continue;
            // debug:
            // printf("truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\n", truth.x, truth.y, truth.w, truth.h);
            float best_iou = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);   // 计算该gt在网格中的第几列
            j = (truth.y * l.h);   // 计算该gt在网格中的第几行
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;    // 将gt的x信息置0处理
            truth_shift.y = 0;    // 将gt的y信息置0处理, 方便后续计算iou时, 不在与位置xy有关,只与wh有关
            //printf("index %d %d\n",i, j);
            // 根据i,j追踪到gt在网格中的位置, 然后在l.outputs中取出该网格中的anchor对应的pred_bbox信息.
            for(n = 0; n < l.n; ++n){
                // 第i,j网格中第n个anchor对应的pred_bbox在outputs内存空间中占据的偏移量.
                // 之后l.outputs[index : index + size]属于该pred_bbox的预测信息(xywh+conf+class_nums_p).
                int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                // 取出anchor对应的pred_bbox坐标信息
                box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                if(l.bias_match){  // 如果是基于anchor进行匹配
                    pred.w = l.biases[2*n];  // 将预测框的wh换成anchor的wh
                    pred.h = l.biases[2*n+1];
                    if(DOABS){
                        pred.w = l.biases[2*n]/l.w;
                        pred.h = l.biases[2*n+1]/l.h;
                    }
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;  // pred_bbox的x和y信息置0处理.
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);  // 计算pred与gt之间的iou
                if (iou > best_iou){
                    best_index = index;  // best_index表示与gt匹配的最大pred_bbox的索引
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);
            // 经过上述for(n = 0; n < l.n; ++n)循环, 能找出每一个gt最佳匹配的anchor的best_index和对应best_iou信息
            // delta_region_box得到的pred_bbox(经gt最佳匹配的anchor和预测(tx,ty,tw,th)转换而来)
            float iou = delta_region_box(truth, l.output, l.biases, best_n, best_index,
                                         i, j, l.w, l.h, l.delta, l.coord_scale);
            if(iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 4];
            // l.delta[best_index+4]记录着正样本的pred_bbox(anchor)的IOU损失对应的梯度信息,
            // 损失表达式: (IOU_truth_r - bij_k_o)^2, IOU_truth_r即为GT与pred_bbox之间的iou, bij_k_o网络输出的iou
            l.delta[best_index + 4] = l.object_scale * (1 - l.output[best_index + 4]) *
                                      logistic_gradient(l.output[best_index + 4]);
            // yolov2采用的是: (iou - l.output[best_index + 4])
            // yolov3采用的是: (1 - l.output[best_index + 4])
            if (l.rescore) {
                l.delta[best_index + 4] = l.object_scale * (iou - l.output[best_index + 4]) *
                                          logistic_gradient(l.output[best_index + 4]);
            }

            if (l.map) class_id = l.map[class_id];
            // 从l.outputs[best_index+5]~l.outputs[best_index+5+classes]是存储这该pred_bbox的预测类别概率值.
            // delta_region_class主要是计算类别损失对应的梯度信息, 只有正样本才计算类别损失.
            delta_region_class(l.output, l.delta, best_index + 5, class_id, l.classes,
                               l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
    #ifndef GPU
    flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    #endif
    // TODO: l->cost的损失计算非常奇怪, 怎么是l.delta*l.delta? 这明显会有问题啊?还是我没看懂?
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Avg_loss: %f, Avg_iou: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  gt_nums: %d\n",
                                            *(l.cost) / count, avg_iou/count, avg_cat/class_count, avg_obj/count,
                                            avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const region_layer l, network_state state)
{
    // l.delta是region层的梯度, state.delta是region层的上一层的梯度信息,
    // state.delta[i*1] += 1 * l.delta[i*1], 相当于直接将l.delta层的梯度复制给上一层.
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i;
    float *const predictions = l.output;
    #pragma omp parallel for
    for (i = 0; i < l.w*l.h; ++i){
        int j, n;
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    /*
       if(!state.train){
       copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
       return;
       }
     */
    flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
    if(l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(l.output_gpu+count, group_size, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
            count += group_size;
        }
    }else if (l.softmax){
        softmax_gpu(l.output_gpu+5, l.classes, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 5);
    }

    float* in_cpu = (float*)xcalloc(l.batch * l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.truths;
        truth_cpu = (float*)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
    //cudaStreamSynchronize(get_cuda_stream());
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    free(cpu_state.input);
    if(!state.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //cudaStreamSynchronize(get_cuda_stream());
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer_gpu(region_layer l, network_state state)
{
    flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
#endif


void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i, j, n, z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w / 2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for (z = 0; z < l.classes + l.coords + 1; ++z) {
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if (z == 0) {
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for (i = 0; i < l.outputs; ++i) {
            l.output[i] = (l.output[i] + flip[i]) / 2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int index = n*l.w*l.h + i;
            for (j = 0; j < l.classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);// , l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask) {
                for (j = 0; j < l.coords - 4; ++j) {
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if (l.softmax_tree) {

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);// , l.w*l.h);
                if (map) {
                    for (j = 0; j < 200; ++j) {
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
                else {
                    int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            }
            else {
                if (dets[index].objectness) {
                    for (j = 0; j < l.classes; ++j) {
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i) {
        for (n = 0; n < l.n; ++n) {
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}
