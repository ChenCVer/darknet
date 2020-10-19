#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

extern int check_mistakes;

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = YOLO;

    l.n = n;           // 当前层分配的anchor数
    l.total = total;   // 总的anchors数
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);  // yolo层的特征通道数
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)xcalloc(1, sizeof(float));   // yolo层损失
    l.biases = (float*)xcalloc(total * 2, sizeof(float));  // 装载anchor
    if(mask) l.mask = mask;  // 需要用到的anchor掩码序列
    else{
        l.mask = (int*)xcalloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));  // 用来存储anchor的的delta?
    l.outputs = h*w*n*(classes + 4 + 1);  // yolo层总共输出元素
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truth_size = 4 + 2;  // 这个truth_size怎么是4+2
    l.truths = l.max_boxes*l.truth_size;    // 200*(4 + 1);
    l.labels = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));  // TODO: ??
    for (i = 0; i < batch * l.w*l.h*l.n; ++i) l.labels[i] = -1;  // 初始化label

    l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));    // 存储误差项
    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));   // 存储输出结果
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;  // 初始化: l.biases
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.output_avg_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "yolo\n");
    srand(time(0));  // 这里怎么还初始化随机种子?

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (l->embedding_output) l->embedding_output = (float*)xrealloc(l->output, l->batch * l->embedding_size * l->n * l->h * l->w * sizeof(float));
    if (l->labels) l->labels = (int*)xrealloc(l->labels, l->batch * l->n * l->h * l->w * sizeof(int));

    if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        CHECK_CUDA(cudaFreeHost(l->output));
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        CHECK_CUDA(cudaFreeHost(l->delta));
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->output_avg_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
    l->output_avg_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
                            // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
                            // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

static inline float fix_nan_inf(float val)
{
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}

static inline float clip_value(float val, const float max_val)
{
    if (val > max_val) {
        //printf("\n val = %f > max_val = %f \n", val, max_val);
        val = max_val;
    }
    else if (val < -max_val) {
        //printf("\n val = %f < -max_val = %f \n", val, -max_val);
        val = -max_val;
    }
    return val;
}

ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw,
                    int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer,
                    IOU_LOSS iou_loss, int accumulate, float max_delta, int *rewritten_bbox)
{
    if (delta[index + 0 * stride] || delta[index + 1 * stride] ||
        delta[index + 2 * stride] || delta[index + 3 * stride]) {
            (*rewritten_bbox)++;
    }

    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    // 获得第j*w+i个cell的第n个bbox在当前特征图的[x,y,w,h]
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    all_ious.iou = box_iou(pred, truth);     // iou
    all_ious.giou = box_giou(pred, truth);   // giou
    all_ious.diou = box_diou(pred, truth);   // diou
    all_ious.ciou = box_ciou(pred, truth);   // ciou
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }  // 避免预测框太小.
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss, yolo v3用
    {
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        // log()函数就像x^(1/2)函数一样,当大框时, tw和th差值小, 但是当小框时, tw和th的差值大一些.
        float tw = log(truth.w*w / biases[2 * n]);
        float th = log(truth.h*h / biases[2 * n + 1]);

        // accumulate delta, 计算误差 delta,乘了权重系数scale=(2-truth.w*truth.h)
        delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;
        delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
    }
    else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;

        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        dw *= exp(x[index + 2 * stride]);
        dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;


        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        if (max_delta != FLT_MAX) {
            dx = clip_value(dx, max_delta);
            dy = clip_value(dy, max_delta);
            dw = clip_value(dw, max_delta);
            dh = clip_value(dh, max_delta);
        }


        if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
        }

        // accumulate delta(累计梯度)
        delta[index + 0 * stride] += dx;
        delta[index + 1 * stride] += dy;
        delta[index + 2 * stride] += dw;
        delta[index + 3 * stride] += dh;
    }

    return all_ious;
}

void averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta)
{

    int classes_in_one_box = 0;
    int c;
    //在一个box里面bbox有多少个类别
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride*c] > 0) classes_in_one_box++;
    }
    //这是边界框的梯度除以box中的物体类别数
    if (classes_in_one_box > 0) {
        delta[box_index + 0 * stride] /= classes_in_one_box;
        delta[box_index + 1 * stride] /= classes_in_one_box;
        delta[box_index + 2 * stride] /= classes_in_one_box;
        delta[box_index + 3 * stride] /= classes_in_one_box;
    }
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers)
{
    int n;
    if (delta[index + stride*class_id]){
        float y_true = 1;
        if(label_smooth_eps) y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
        float result_delta = y_true - output[index + stride*class_id];
        if(!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*class_id] = result_delta;
        //delta[index + stride*class_id] = 1 - output[index + stride*class_id];

        if (classes_multipliers) delta[index + stride*class_id] *= classes_multipliers[class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    if (focal_loss) {
        // TODO: 这里备注下关于focal loss的计算,设定label的one-hot形式其y_k=1,
        //  我们知道的交叉熵损失定义: J=-∑y_ilog(p_i)=-log(p_k);
        //  Focal loss: FL=-∑(y_i)*[(1-p_i)^γ]*[log(p_i)]=-[(1-p_k)^γ]*[log(p_k)]
        //  对于Loss关于p的导数:
        //  交叉熵: ∂FLoss/∂p = -1/p;
        //  FL-loss: ∂FLoss/∂p = -γ*[(1-p)^(γ-1)]*log(p)-[(1-p)^γ]*(1/p);
        //  当γ=2时, ∂FLoss/∂p = -2*(1-p)*log(p)-[(1-p)^2]*(1/p);
        //  我们知道, p=softmax(x)情况下, ∂p/∂x_i: 当i=k时, ∂p/∂x_i=p_i(1-p_i),
        //  当i≠k时,∂p/∂x_i=p_i*p_k,
        //  Note: 从下面grad变量看出, ∂FLoss/∂p被乘以了一个p.也即: grad = (∂Loss/∂p)*p,
        //  结合: ∂FLoss/∂x = (∂FLoss/∂p) * (∂p/∂x)
        //  如果grad中的惩罚因子来自于∂p/∂x中的p_i, 则这里(∂p/∂x)就可以直接写成ce+softmax的联合反向负梯度形式: (t-p)
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;  // sigmoid函数输出结果
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        // http://blog.csdn.net/linmingan/article/details/77885832
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);  // (∂FLoss/∂p) x p
        // https://github.com/unsky/focal-loss
        // float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);

        for (n = 0; n < classes; ++n) {
            // TODO: darknet为了代码统一, 这里都统一写成了ce+softmax的backward()形式,
            //  实际上, 如果按部就班写的话: grad只是(∂FLoss/∂p), 下面这句delta[index + stride*n]为softmax()的反向传播数值
            //  但是事实是grad被乘以一个p后, 这样delta[index + stride*n]直接就跟ce+softmax的形式一模一样,搞得人摸不着头脑.
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);
            delta[index + stride*n] *= alpha*grad;

            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            float y_true = ((n == class_id) ? 1 : 0);
            if (label_smooth_eps)
                y_true = y_true * (1 - label_smooth_eps) + 0.5 * label_smooth_eps;
            float result_delta = y_true - output[index + stride*n];  // 分类损失梯度.这里是交叉熵损失梯度
            if (!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*n] = result_delta;

            if (classes_multipliers && n == class_id) delta[index + stride*class_id] *= classes_multipliers[class_id];
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

int compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh)
{
    int j;
    for (j = 0; j < classes; ++j) {
        //float prob = objectness * output[class_index + stride*j];
        float prob = output[class_index + stride*j];
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief 计算某个矩形框中某个参数在l.output中的索引。一个矩形框包含了x,y,w,h,c,C1,C2...,Cn信息,
 *        前四个用于定位,第五个为矩形框含有物体的置信度信息c,即矩形框中存在物体的概率为多大,而C1到Cn
 *        为矩形框中所包含的物体分别属于这n类物体的概率。本函数负责获取该矩形框首个定位信息也即x值在
 *        l.output中索引、获取该矩形框置信度信息c在l.output中的索引、获取该矩形框分类所属概率的首个
 *        概率也即C1值的索引,具体是获取矩形框哪个参数的索引,取决于输入参数entry的值,由于l.output的
 *        存储方式,当entry=0时,就是获取矩形框x参数在l.output中的索引;当entry=4时,就是获取矩形框
 *        置信度信息c在l.output中的索引;当entry=5时,就是获取矩形框首个所属概率C1在l.output中的索引.
 * @param l: 当前yolo层
 * @param batch: 当前照片是整个batch中的第几张,因为l.output中包含整个batch的输出,所以要定位某张训练图片
 *               输出的众多网格中的某个矩形框,当然需要该参数.
 * @param location 这个参数,说实话,感觉像个鸡肋参数,函数中用这个参数获取n和loc的值,这个n就是表示网格中
 *                 的第几个预测矩形框(比如每个网格预测5个矩形框,那么n取值范围就是从0~4,loc就是某个
 *                 该预测框位于cell中的(j, i)位置.
 * @param entry 切入点偏移系数(通道偏移), entry每增加1, 相当于偏移一个特征图.
 * @details l.output中存储了整个batch的训练输出,每张训练图片都会输出l.out_w*l.out_h个网格,每个网格会预测l.n
 *          个矩形框,每个矩形框含有l.classes+l.coords+1个参数,而最后一层的输出通道数为l.n*(l.classes+l.coords+1)
 *          ,可以想象下最终输出的三维张量是个什么样子的。展成一维数组存储时,l.output可以首先分成batch个大段,每个大段存
 *          储了一张训练图片的所有输出;进一步细分,取其中第一大段分析,该大段中存储了第一张训练图片所有输出网格预测的矩形框信
 *          息,每个网格预测了l.n个矩形框,存储时,l.n个矩形框是分开存储的,也就是先存储所有网格中的第一个矩形框,而后存储所
 *          有网格中的第二个矩形框,依次类推,如果每个网格中预测5个矩形框,则可以继续把这一大段分成5个中段。继续细分,5个中段
 *          中取第一个中段来分析,这个中段中按行(有l.out_w*l.out_h个网格,按行存储)依次存储了这张训练图片所有输出网格中
 *          的第一个矩形框信息,要注意的是,这个中段存储的顺序并不是挨个挨个存储每个矩形框的所有信息,而是先存储所有矩形框的x,
 *          而后是所有的y,然后是所有的w,再是h,c,最后的的概率数组也是拆分进行存储,并不是一下子存储完一个矩形框所有类的概率,
 *          而是先存储所有网格所属第一类的概率,再存储所属第二类的概率,具体来说这一中段首先存储了l.out_w*l.out_h个x,然后
 *          是l.out_w*l.out_c个y,依次下去,最后是l.out_w*l.out_h个C1(属于第一类的概率,用C1表示,下面类似),
 *          l.out_w*l.outh个C2,...,l.out_w*l.out_c*Cn(假设共有n类),所以可以继续将中段分成几个小段,依次为x,y,w,h,c,
 *          C1,C2,...Cn小段,每小段的长度都为l.out_w*l.out_c.现在回过来看本函数的输入参数,batch就是大段的偏移数(从第几个
 *          大段开始,对应是第几张训练图片),由location计算得到的n就是中段的偏移数(从第几个中段开始,对应是第几个矩形框),
 *          entry就是小段的偏移数(从几个小段开始,对应具体是那种参数,x,c还是C1, 也就是通道偏移),而loc则是最后的定位(cell位置),
 *          前面确定好第几大段中的第几中段中的第几小段的首地址,loc就是从该首地址往后数loc个元素,得到最终定位
 *          某个具体参数(x或c或C1)的索引值,比如l.output中存储的数据如下所示(这里假设只存了一张训练图片的输出,
 *          因此batch只能为0;并假设l.out_w=l.out_h=2,l.classes=2)：
 *          xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2-#-xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2,
 *          n=0则定位到-#-左边的首地址(表示每个网格预测的第一个矩形框),n=1则定位到-#-右边的首地址(表示每个网格预测的第二个矩形框)
 *          entry=0,loc=0获取的是x的索引,且获取的是第一个x也即l.out_w*l.out_h个网格中第一个网格中第一个矩形框x参数的索引;
 *          entry=4,loc=1获取的是c的索引,且获取的是第二个c也即l.out_w*l.out_h个网格中第二个网格中第一个矩形框c参数的索引;
 *          entry=5,loc=2获取的是C1的索引,且获取的是第三个C1也即l.out_w*l.out_h个网格中第三个网格中第一个矩形框C1参数的索引;
 *          如果要获取第一个网格中第一个矩形框w参数的索引呢？如果已经获取了其x值的索引,显然用x的索引加上3*l.out_w*l.out_h即可获取到,
 *          这正是delta_region_box()函数的做法;
 *          如果要获取第三个网格中第一个矩形框C2参数的索引呢？如果已经获取了其C1值的索引,显然用C1的索引加上l.out_w*l.out_h即可获取到,
 *          这正是delta_region_class()函数中的做法;
 *          由上可知,entry=0时,即偏移0个小段,是获取x的索引;entry=4,是获取自信度信息c的索引;entry=5,是获取C1的索引.
 */
static int entry_index(layer l, int batch, int location, int entry)
{   // location = n*l.w*l.h, 这个entry为特征图的id(第几张特征图)
    // 前4张特征图分别表示xywh.
    int n =   location / (l.w*l.h);  // n表示第几个anchor
    int loc = location % (l.w*l.h);  // loc表示第几个格子中
    // 第batch张图的第n个anchor中第enrty张特征图上的第loc个格子.
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}


void forward_yolo_layer(const layer l, network_state state)
{   // l.outputs = h*w*num_anchors*(num_classes+xywh+c)
    int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            // 这里的index通过函数entry_index()后表达的意思是(配合紧跟在后面的activate_array()):
            // 第n(0~l.n-1)个anchor的所有预测量在l.output上的起始位置(因为entry为0, 也就是第0层特征图位置),
            // 举个例子: 如果yolo层有3个负责的anchor.则l.output的存储信息为[anchor_1的所有预测值, anchor_2
            // 的所有预测值, anchor_3的所有预测值].
            // 其中每个anchor所有的预测值所占的内存空间大小为: l.w*l.h*(4+l.classes+1), 现在函数entry_index()
            // 中对于entry传入参数的是0, 也即第0个特征层上.
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // 对预测框的的x和y进行sigmoid()操作,从这里可以看出anchor所有预测值在l.output上内存分布:
            // 假如: l.w=l.h=2,l.classes=2,则有: [xxxxyyyywwwwhhhhcccc1c1c1c1c2c2c2c2]
            // 其中: xxxx表示第n个预测框分别在第1/2/3/4个网格中的预测量x.
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            // TODO: ??(l.output + index)[N*1] = (l.output + index)[N*1] * l.scale_x_y -0.5*(l.scale_x_y - 1),
            //  这个l.scale_x_y起什么作用呢?
            scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);
            // 这个index表示的含义为(): 第n(0~l.n-1)个anchor的所有预测值中, 包含的所有conf和class_score部分,
            // 在l.output上的起始位置.
            index = entry_index(l, b, n*l.w*l.h, 4);
            // 对conf和class_score执行sigmoid()操作.注意yolo v2中, class_score经过的是softmax操作.
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // delta is zeroed
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));  // l.detla梯度清零
    if (!state.train) return;

    for (i = 0; i < l.batch * l.w*l.h*l.n; ++i) l.labels[i] = -1;  // TODO: ??这个label到底是啥?
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;  // 记录着每个pred_bbox输出的confidence值.
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;   // 损失清零
    // 遍历每张图中的每个格子(j, i)中的第n的anchor
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    // 获取第b张图片对应的特征图中的(j, i)网格中的第n个bbox的第4+1张特征图~第25张(voc为例, 20个类别).
                    const int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                    // 获取第b张图片对应的特征图中的(j, i)网格中的第n个bbox的confidence值下标.
                    const int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    // 获取第b张图片对应的特征图中的(j, i)网格中的第n个bbox的xywh起始下标位置.
                    const int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    const int stride = l.w*l.h;
                    // 计算第j*w+i个cell第n个bbox在当前特征图上的相对位置[x,y](相对于特征图),在网络输入图片上的
                    // 相对宽度,高度[w,h](相对于网络输入图). state.net.w和 state.net.h记录的是网络输入尺寸.
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h,
                                            state.net.w, state.net.h, l.w*l.h);
                    float best_match_iou = 0;
                    int best_match_t = 0;
                    float best_iou = 0;
                    int best_t = 0;
                    for (t = 0; t < l.max_boxes; ++t) {  // 遍历图片中的每个gt
                        // 获取第t个gt的xywh值.
                        box truth = float_to_box_stride(state.truth + t*l.truth_size + b*l.truths, 1);
                        if (!truth.x) break;  // 遍历完所有的有效gt,即时退出
                        int class_id = state.truth[t*l.truth_size + b*l.truths + 4];  // 获取gt对应的类别cls_id
                        if (class_id >= l.classes || class_id < 0) {
                            printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            printf("\n truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                            if (check_mistakes) getchar();
                            continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
                        }

                        float objectness = l.output[obj_index];  // 获取confidence值

                        if (isnan(objectness) || isinf(objectness)) l.output[obj_index] = 0;
                        // 如果pred_bbox的预测的所有类别概率中, 如果某一类的类别概率超过了0.25, 则设置class_id_match=1
                        int class_id_match = compare_yolo_class(l.output, l.classes, class_index,
                                                                l.w*l.h, objectness, class_id, 0.25f);
                        // 计算pred_bbox与gt的iou
                        float iou = box_iou(pred, truth);
                        // 这个地方和pje的darknet实现不太一样,多了一个class_id_match=1的限制,即预测bbox的某一类概率必须大于0.25
                        if (iou > best_match_iou && class_id_match == 1) {
                            best_match_iou = iou;
                            best_match_t = t;
                        }  // TODO: ?? 这个best_match_iou和best_iou有什么区别?
                        if (iou > best_iou) {
                            best_iou = iou;  // best_iou记录这每个pred_bbox匹配的最佳gt, 基于iou最大原则.
                            best_t = t;      // 记录匹配最佳的gt的编号
                        }
                    }
                    // 通过上面的for循环, 找出了对于每个pred_bbox来说, 最佳的gt
                    avg_anyobj += l.output[obj_index];
                    // 与yolov1 v2相似,初始时将pred bbox都当做noobject(负样本), 计算其confidence梯度,
                    // 不过这里多了一个平衡系数.
                    // TODO: 首先l.delta[obj_index]表示的是误差项δ=∂Loss/∂net_j,这个net_j表示的是加权输入值.
                    //  举个例子, 假如用a_l-1表示第l-1层的神经元的输出, 则有a_l-1 = f(net_l-1), 其中这个f表示的
                    //  激活函数.进而有: 对于卷积层来说net_l = conv(w_l, a_l-1)+w_b, 具体到这个sigmoid层来说,
                    //  a_l = sigmoid(net_l),这个a_l就是概率值p, 注意这个net_l其实就是a_l-1. 总的关系就是:
                    //  a_l-1 --> 恒等变换y=x --> net_l --> sigmoid() --> a_l ---> bce() -->loss
                    //  因此, 有:
                    //  δ=∂Loss/∂net_l=(∂Loss/∂a_l)*(∂a_l/∂net_l), 置信度损失用的是二分交叉熵损失,
                    //  参考: https://www.cnblogs.com/nowgood/p/sigmoidcrossentropy.html
                    //  (∂Loss/∂a_l)=(p-label)/[p(1-p)], (∂a_l/∂net_l)=p(1-p)
                    //  因此,(∂Loss/∂net_l)=p-label, darknet统一采用的负梯度, +=形式.也即 +(-grandient).
                    //  因为label始终是0, 因此这里是: (0 - l.output[obj_index])
                    l.delta[obj_index] = l.cls_normalizer * (0 - l.output[obj_index]);
                    // best_match_iou大于阈值则说明pred_box有物体, 在yolov3中阈值ignore_thresh=.5
                    // 这里需要注意一个事情, best_match_iou > l.ignore_thresh,可知,该pred_bbox不一定是正样本
                    // 暂时把他当做了忽略样本看待.
                    if (best_match_iou > l.ignore_thresh) {
                        // TODO: 这个l.objectness_smooth不理解作者想干什么?
                        if (l.objectness_smooth) {
                            const float delta_obj = l.cls_normalizer * (best_match_iou - l.output[obj_index]);
                            if (delta_obj > l.delta[obj_index])
                                l.delta[obj_index] = delta_obj;
                        }
                        // l.delta[obl_index]位置需要清零处理.
                        else l.delta[obj_index] = 0;
                    }
                    else if (state.net.adversarial) {  // TODO: state.net.adversarial这个不知道是什么?
                        int stride = l.w*l.h;
                        float scale = pred.w * pred.h;
                        if (scale > 0) scale = sqrt(scale);
                        l.delta[obj_index] = scale * l.cls_normalizer * (0 - l.output[obj_index]);
                        int cl_id;
                        int found_object = 0;
                        for (cl_id = 0; cl_id < l.classes; ++cl_id) {
                            if (l.output[class_index + stride*cl_id] * l.output[obj_index] > 0.25) {
                                l.delta[class_index + stride*cl_id] = scale * (0 - l.output[class_index + stride*cl_id]);
                                found_object = 1;
                            }
                        }
                        if (found_object) {
                            // don't use this loop for adversarial attack drawing
                            for (cl_id = 0; cl_id < l.classes; ++cl_id)
                                if (l.output[class_index + stride*cl_id] * l.output[obj_index] < 0.25)
                                    l.delta[class_index + stride*cl_id] = scale * (1 - l.output[class_index + stride*cl_id]);

                            l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]);
                            l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]);
                            l.delta[box_index + 2 * stride] += scale * (0 - l.output[box_index + 2 * stride]);
                            l.delta[box_index + 3 * stride] += scale * (0 - l.output[box_index + 3 * stride]);
                        }
                    }
                    // 如果pred bbox为完全预测正确样本,在yolov3完全预测正确样本的阈值truth_thresh=1.
                    //这个参数在cfg文件中值为1.这个条件语句永远不可能成立
                    // TODO: 下面这个if如果条件成立, 就很有可能出现这种情况: 假如当前正在遍历的cell(1,2)中的第2个pred_bbox,
                    //  然后遍历图片中所有的gt, 如果这个pred_bbox对应的best_iou所对应的gt(准确说是gt中心位置)并不在该cell(1,2)
                    //  内部, 这就打破了: 处在某个cell里面的gt只能由该cell里面的pred_bbox来负责预测了限制.
                    //  这样就出现了一个严重的问题, 网络预测的x和y方面的偏移量是0~1之间, 也就是说, cell里面的anchor根据网络预测量
                    //  无法将这个anchor与他负责的gt之间的差距拉近(因为这个gt在别的cell里面), 导致最后网络都可能处于无法收敛的状态.
                    if (best_iou > l.truth_thresh) {
                        // 作者在yolo v3论文中的第4节提到了这部分。
                        // 作者尝试Faster-RCNN中提到的双IOU策略, 当Anchor与GT的IoU大于0.7时,
                        // 该Anchor被算作正样本计入损失中,但训练过程中并没有产生好的结果,所以最后放弃了.
                        // 如果要模仿faster rcnn, 这里要将l.truth_thresh改成0.7即可.
                        // Note: 这里是每一个anchor与所有gt进行匹配, 基于IOU最大原则, 找到每个anchor对应的最佳gt.
                        const float iou_multiplier = best_iou*best_iou;
                        if (l.objectness_smooth)
                            l.delta[obj_index] = l.cls_normalizer * (iou_multiplier - l.output[obj_index]);
                        else
                            l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);  // 正样本的conf损失.

                        // 获得best_iou对应GT bbox的class的index
                        int class_id = state.truth[best_t*l.truth_size + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id]; //yolov3 yolo层中map=0, 不参与计算
                        // 计算正样本的分类损失及梯度
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0,
                                         l.focal_loss, l.label_smooth_eps, l.classes_multipliers);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        if (l.objectness_smooth)
                            l.delta[class_index + stride*class_id] = class_multiplier *
                                    (iou_multiplier - l.output[class_index + stride*class_id]);
                        // 获取最匹配的gt的xywh
                        box truth = float_to_box_stride(state.truth + best_t*l.truth_size + b*l.truths, 1);
                        // 计算位置损失及梯度
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w,
                                       state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h,
                                       l.iou_normalizer * class_multiplier, l.iou_loss, 1,
                                       l.max_delta, state.net.rewritten_bbox);
                        (*state.net.total_bbox)++;  // total_bbox记录正样本数量.
                    }
                }
            }
        }
        // 遍历每一张图片中的所有gt, 注意, 制作yolo的标签信息时候, 已经将gt的xywh等除以输入尺度了.
        // TODO: YOLO算法一般默认一个cell里面只有一个gt, 但是如果一张图片里面gt比较多, 最后难免出现一个cell中会出现
        //  多个gt的情况, 这样就会出现一个所谓的标签重写问题.具体来说: 假如cell(1,1)里面有两个gt, 我在遍历gt_1的时候,
        //  anchor_2是最佳匹配, 后续接着会进行一些列分类损失和回归损失计算等, 当我遍历到gt_2(如果gt_2和gt_1的wh差不多)
        //  的时候, 又和anchor_2匹配上了. 后面接着进行的一系列损失计算会覆盖之前gt_1的匹配结果, 也即anchor_2现在是负责
        //  gt_2而不是gt_1了. 导致gt_1部分成了背景区域.后续yolo-poly针对此问题有相应的策略, 感兴趣请阅读:
        //  https://www.zybuluo.com/huanghaian/note/1712318
        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t*l.truth_size + b*l.truths, 1);  // 获取gt的xywh
            if (!truth.x) break;  // continue;
            // printf("truth: %d, box: %f\t%f\t%f\t%f\n", t, truth.x, truth.y, truth.w, truth.h);
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                char buff[256];
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list",
                    truth.x, truth.y, truth.w, truth.h);
                system(buff);
            }
            int class_id = state.truth[t*l.truth_size + b*l.truths + 4];  // 获取gt对应的cls_id
            if (class_id >= l.classes || class_id < 0) continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value

            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);  // 获取gt所在当前特征图的(j, i)位置
            j = (truth.y * l.h);  //
            box truth_shift = truth;  //将truth_shift的box位置移动到0,0, 计算iou时不考虑gt的位置.
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {  // 遍历所有的anchor, 找到与当前第t个gt的IOU最大的anchor
                box pred = {0};
                pred.w = l.biases[2 * n] / state.net.w;  // l.biases里面存的是anchor信息,
                pred.h = l.biases[2 * n + 1] / state.net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;  // 记录与gt匹配最佳的anchor的IOU
                    best_n = n;      // 记录对应的编号信息
                }
            }
            // printf("best_n: %d\n", best_n);

            // 这里是为了验证, 对于第t个gt, 与他匹配最佳的anchor的编号, 是否是由该层的anchor预测的.
            int mask_n = int_index(l.mask, best_n, l.n);  // l.n即为该层anchor的个数.
            // TODO: 通过打印输出发现, 同一个gt, 在3个预测层所匹配的最佳anchor编号best_n不会变, 也就是说
            //  虽然YOLO v3有三个yolo层, 每次经过yolo层的时候, 每个gt都会计算一遍得出最佳匹配anchor, 但是
            //  每个gt在三个yolo层下算出的best_n不会变.
            if (mask_n >= 0) {  // 如果gt匹配到了该层负责的anchor
                int class_id = state.truth[t*l.truth_size + b*l.truths + 4];  // 获取第t个gt对应的类别信息
                if (l.map) class_id = l.map[class_id];
                // 获得与第t个gt匹配最佳的anchor的编号index: 第b张图的第(j, i)网格位置的第mask_n个anchor
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                // 计算位置损失, 边界框回归损失,支持: MSE, IOU Loss, GIOU Loss, DIOU Loss, CIOU Loss.
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h,
                                               state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h),
                                               l.w*l.h, l.iou_normalizer * class_multiplier,
                                               l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox);

                (*state.net.total_bbox)++;  // 记录所有正样本个数.

                const int truth_in_index = t*l.truth_size + b*l.truths + 5;
                const int track_id = state.truth[truth_in_index];
                const int truth_out_index = b*l.n*l.w*l.h + mask_n*l.w*l.h + j*l.w + i;
                l.labels[truth_out_index] = track_id;

                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;

                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;

                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;

                // 这里是获得与gt最匹配的anchor对应的pred_bbox, 它预测的conf值的下标索引
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];  // 统计所用正样本的conf值
                if (l.objectness_smooth){
                    float delta_obj = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);
                    if (l.delta[obj_index] == 0) l.delta[obj_index] = delta_obj;
                }
                else
                    // 计算正样本的置信度损失.
                    l.delta[obj_index] = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);
                // 获得与gt最匹配的anchor对应的pred_bbox, 它预测的class_score值的起始索引位置
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                // 计算分类损失
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat,
                                 l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                //printf(" label: class_id = %d, truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", class_id, truth.x, truth.y, truth.w, truth.h);
                //printf(" mask_n = %d, l.output[obj_index] = %f, l.output[class_index + class_id] = %f \n\n", mask_n, l.output[obj_index], l.output[class_index + class_id]);

                ++count;  // 正样本数量
                ++class_count;  // 这个难道不也是正样本?
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }

            // iou_thresh, 遍历每一个anchor, 下面这个过程和上面一样，不过多约束了一个iou_thresh
            for (n = 0; n < l.total; ++n) {
                int mask_n = int_index(l.mask, n, l.n);  // 通过这个限制, 只有对当前规格的特征层负责的anchor, 才有mask_n>0成立.
                // 这里是考察当前层的anchor中, 除了与gt匹配最佳的anchor之外, 其他的anchor的情况.
                if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) {
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.net.w;
                    pred.h = l.biases[2 * n + 1] / state.net.h;
                    // 计算第t个gt与anchor的iou值
                    float iou = box_iou_kind(pred, truth_shift, l.iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU

                    // 如果第t个gt与anchor的iou大于设定值, 则该anchor也负责预测第t个gt.
                    if (iou > l.iou_thresh) {
                        int class_id = state.truth[t*l.truth_size + b*l.truths + 4];  // 获取gt的类别编号
                        if (l.map) class_id = l.map[class_id];
                        // 获得与第t个gt匹配的anchor(非最佳)的编号index: 第b张图的第(j, i)网格位置的第mask_n个anchor
                        int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        // 计算边界框损失
                        ious all_ious = delta_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h,
                                                       state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h),
                                                       l.w*l.h, l.iou_normalizer * class_multiplier,
                                                       l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox);
                        (*state.net.total_bbox)++;  // 正样本数量+1

                        // range is 0 <= 1
                        tot_iou += all_ious.iou;
                        tot_iou_loss += 1 - all_ious.iou;
                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        tot_giou_loss += 1 - all_ious.giou;

                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;

                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;
                        // 获取anchor对应的pred_bbox的conf值
                        int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                        avg_obj += l.output[obj_index];  // 记录iou值
                        if (l.objectness_smooth) {
                            float delta_obj = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);
                            if (l.delta[obj_index] == 0) l.delta[obj_index] = delta_obj;
                        }
                        else
                            // confidence损失
                            l.delta[obj_index] = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);

                        // 获取类别预测值其实索引编号
                        int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                        // 计算分类损失
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                        ++count;
                        ++class_count;
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
                    }
                }
            }
        }

        // 遍历每一个pred_bbox,
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    // 找到该anchor(cell(j,i)的第n个anchor)对应pred_bbox的预测值信息在l.output上的起始位置编号.
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    // 找到该anchor(cell(j,i)的第n个anchor)对应pred_bbox的分类预测值信息在l.output上的起始位置编号.
                    int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                    const int stride = l.w*l.h;  //特征图的大小
                    // 对梯度进行平均, 注意平均的梯度是边界框的梯度!
                    averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
                }
            }
        }
    }

    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;

    int stride = l.w*l.h;
    float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));  // 计算IOU loss所需
    memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));  // 把l.delta拷贝给no_iou_loss_delta
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    // 遍历每一个pred_bbox, index为pred_bbox的预测信息在l.output上的起始位置
                    int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    // 对应的iou_loss_delta置位0.
                    no_iou_loss_delta[index + 0 * stride] = 0;
                    no_iou_loss_delta[index + 1 * stride] = 0;
                    no_iou_loss_delta[index + 2 * stride] = 0;
                    no_iou_loss_delta[index + 3 * stride] = 0;
                }
            }
        }
    }
    float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
    free(no_iou_loss_delta);

    float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);   // default, yolov3原始损失.
    float iou_loss = loss - classification_loss;

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);   // default, yolov3原始损失.
    }
    else {
        // Always compute classification loss both for iou + cls loss and for logging with mse loss
        // TODO: remove IOU loss fields before computing MSE on class
        //   probably split into two arrays
        if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
        }
        else {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
        }
        *(l.cost) = avg_iou_loss + classification_loss;
    }

    loss /= l.batch;  // 这个loss即为原版YOLO V3 loss.
    classification_loss /= l.batch;
    iou_loss /= l.batch;

    fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
        (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.cls_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        classification_loss, iou_loss, loss);

    // printf("-----------------------------------------\n");
}

void backward_yolo_layer(const layer l, network_state state)
{
   // state.delta[i*1] += 1* l.delta[i*1]; state.delta为上一次层的误差项图
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w / netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h / neth;
    for (i = 0; i < n; ++i) {

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}

/*
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
*/

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for(n = 0; n < l.n; ++n){
        for (i = 0; i < l.w*l.h; ++i) {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

int yolo_num_detections_batch(layer l, float thresh, int batch)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, batch, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    // This snippet below is not necessary
    // Need to comment it in order to batch processing >= 2 images
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

int get_yolo_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch)
{
    int i,j,n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, batch, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, batch, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, batch, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, batch, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    if (l.embedding_output) {
        layer le = state.net.layers[l.embedding_layer_id];
        cuda_pull_array_async(le.output_gpu, l.embedding_output, le.batch*le.outputs);
    }

    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            if (l.scale_x_y != 1) scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        if (l.mean_alpha && l.output_avg_gpu) mean_array_gpu(l.output_gpu, l.batch*l.outputs, l.mean_alpha, l.output_avg_gpu);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, state.net.loss_scale, l.delta_gpu, 1, state.delta, 1);
}
#endif
