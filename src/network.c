#include "darknet.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "conv_lstm_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "reorg_old_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "scale_channels_layer.h"
#include "sam_layer.h"
#include "yolo_layer.h"
#include "gaussian_yolo_layer.h"
#include "upsample_layer.h"
#include "parser.h"

load_args get_base_args(network *net)
{
    load_args args = { 0 };
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

int64_t get_current_iteration(network net)
{
    return *net.cur_iteration;
}

int get_current_batch(network net)
{   // net.seen表示目前已经读入的图片张数(网络已经处理的图片张数)
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

/*
void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        //if(net.gpu_index >= 0) update_network_gpu(net);
    #endif
}
*/

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
#ifdef GPU
        layer l = net->layers[i];
        if (l.state_gpu) {
            fill_ongpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if (l.h_gpu) {
            fill_ongpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
#endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_seq_subdivisions(network net)
{
    int sequence_subdivisions = net.init_sequential_subdivisions;

    if (net.num_steps > 0)
    {
        int batch_num = get_current_batch(net);
        int i;
        for (i = 0; i < net.num_steps; ++i) {
            if (net.steps[i] > batch_num) break;
            sequence_subdivisions *= net.seq_scales[i];
        }
    }
    if (sequence_subdivisions < 1) sequence_subdivisions = 1;
    if (sequence_subdivisions > net.subdivisions) sequence_subdivisions = net.subdivisions;
    return sequence_subdivisions;
}

int get_sequence_value(network net)
{
    int sequence = 1;
    if (net.sequential_subdivisions != 0) sequence = net.subdivisions / net.sequential_subdivisions;
    if (sequence < 1) sequence = 1;
    return sequence;
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);   // 网络当前已经看完的batch数目, 也即迭代次数
    int i;
    float rate;
    // 这个burn_in也即warm_up策略. batch_num也即当前迭代次数
    if (batch_num < net.burn_in)
        return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        case SGDR:
        {
            int last_iteration_start = 0;
            int cycle_size = net.batches_per_cycle;
            while ((last_iteration_start + cycle_size) < batch_num)
            {
                last_iteration_start += cycle_size;
                cycle_size *= net.batches_cycle_mult;
            }
            rate = net.learning_rate_min +
                0.5*(net.learning_rate - net.learning_rate_min)
                * (1. + cos((float)(batch_num - last_iteration_start)*3.14159265 / cycle_size));

            return rate;
        }
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
            return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case GAUSSIAN_YOLO:
            return "Gaussian_yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case SCALE_CHANNELS:
            return "scale_channels";
        case SAM:
            return "sam";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network make_network(int n)  //为network结构体分配内存空间
{
    network net = {0};  // 这句话表示为结构体network所有变量首先进行初始化,所有元素初始化为0, NULL也是0.
    net.n = n;  // 网络层数
    net.layers = (layer*)xcalloc(net.n, sizeof(layer));         // 为每个网络层在堆上开辟内存空间
    net.seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));  // 已被训练的图片数
    net.cur_iteration = (int*)xcalloc(1, sizeof(int));   // 当前迭代次数
    net.total_bbox = (int*)xcalloc(1, sizeof(int));
    net.rewritten_bbox = (int*)xcalloc(1, sizeof(int));
    *net.rewritten_bbox = *net.total_bbox = 0;
#ifdef GPU
    net.input_gpu = (float**)xcalloc(1, sizeof(float*));
    net.truth_gpu = (float**)xcalloc(1, sizeof(float*));

    net.input16_gpu = (float**)xcalloc(1, sizeof(float*));
    net.output16_gpu = (float**)xcalloc(1, sizeof(float*));
    net.max_input16_size = (size_t*)xcalloc(1, sizeof(size_t));
    net.max_output16_size = (size_t*)xcalloc(1, sizeof(size_t));
#endif
    return net;
}


/*
** 前向计算网络net每一层的输出
** state用来标记当前网络的状态,
** 遍历net的每一层网络,从第0层到最后一层,逐层计算每层的输出
*/
void forward_network(network net, network_state state)
{
    // 网络的工作空间, 指的是所有层中占用运算空间最大的那个层的workspace_size,
    // 因为实际上在GPU或CPU中某个时刻只有一个层在做前向或反向运算
    state.workspace = net.workspace;
    int i;
    // 遍历所有层,从第一层到最后一层,逐层进行前向传播,网络共有net.n层
    for(i = 0; i < net.n; ++i){
        // 当前正在进行第i层的处理
        state.index = i;
        // 获取当前层
        layer l = net.layers[i];
        // 如果当前层的l.delta已经动态分配了内存, 则调用fill_cpu()函数将其所有元素初始化为0
        if(l.delta && state.train){  // l.delta不为NULL, 且为训练状态.
            // 第一个参数为l.delta的元素个数, 第二个参数为初始化值, 为0
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);  // l.delta[i*1] *= 0.
        }
        // double time = get_time_point();
        // 前向传播: 完成当前层前向推理
        l.forward(l, state);  // 函数指针, 实现多态.
        // 完成某一层的推理时, 置网络的输入为当前层的输出(这将成为下一层网络的输入), 注意此处更改的是state, 而非原始的net
        // printf("%d - Predicted in %lf milli-seconds.\n", i, ((double)get_time_point() - time) / 1000);
        state.input = l.output;  // l.output记录网络某一层的输出结果, 网络某一层的输出即为下一层的输入

        /*
        // debug:
        float avg_val = 0;
        int k;
        for (k = 0; k < l.outputs; ++k) avg_val += l.output[k];
        printf(" i: %d - avg_val = %f \n", i, avg_val / l.outputs);
        */
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);  // rate即为学习率: learning_rate
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){  // 如果需要更新权重的话.
            l.update(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    return sum/count;
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

/*
** 反向计算网络net每一层的敏感度图(误差项δ),并进而计算每一层的权重、偏置更新值,
** 最后完成每一层权重与偏置更新.
** 流程: 遍历net的每一层网络,从最后一层到第一层(此处所指的第一层不是指输入层,
** 而是与输入层直接相连的第一层隐含层).
*/
void backward_network(network net, network_state state)
{
    int i;
    // 在进行反向传播之前先保存一下原来的net信息
    float *original_input = state.input;
    float *original_delta = state.delta;
    state.workspace = net.workspace;
    for(i = net.n-1; i >= 0; --i){  // n-1~0, 总共107层.
        state.index = i;  // 标志参数, 当前网络的活跃层 
        // i = 0时,也即已经到了网络的第1层(或者说第0层),就是直接与输入层相连的第一层隐含层
        // (注意不是输入层,我理解的输入层就是指输入的图像数据,严格来说,输入层不算一层网络,因为输入层没有训练
        // 参数,也没有激活函数),这个时候,不需要else中的赋值,1)对于第1层来说,其前面已经没有网络层了(输入层不算),
        // 因此没有必要再计算前一层的参数,故没有必要在获取上一层; 2)第一层的输入就是图像输入,也即整个net最原始的
        // 输入,在开始进行反向传播之前,已经用orig变量保存了最为原始的net,所以net.input就是第一层的输入,不需要
        // 通过net.input=prev.output获取上一层的输出作为当前层的输入; 3)同1),第一层之前已经没有层了,也就不需要
        // 计算上一层的delta,即不需要再将net.delta链接到prev.delta,此时进入到l.backward()中后,net.delta就是
        // NULL(可以参看network.h中关于delta的注释),也就不会再计算上一层的敏感度了(比如卷积神经网络中的
        // backward_convolutional_layer()函数)
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }
        else{
            // 获取net.layers[i]的上一层net.layer[i-1].
            layer prev = net.layers[i-1];
            // prev.output也即a_l-1, 上一层的输出值a_l-1作为当前层的输入, 下面l.backward()会用到
            state.input = prev.output;
            // 上一层的敏感度图δ_l-1, 敏感度也即误差项.
            state.delta = prev.delta;
        }
        // 置网络当前活跃层为当前层, 即第i层
        layer l = net.layers[i];
        if (l.stopbackward) break;
        if (l.onlyforward) continue;
        // 反向计算第i层的敏感度图、权重及偏置更新值,并更新权重、偏置(同时会计算上一层的敏感度图,
        // 存储在net.delta中,但是还差一个环节: 乘上上一层输出对加权输入的导数,也即上一层激活函数
        // 对加权输入的导数)
        l.backward(l, state);
    }
}

float train_network_datum(network net, float *x, float *y)
{
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net, x, y);
#endif
    network_state state={0};  // 用network_state结构体记录网络训练过程中forward()和backbard()需要的信息.
    *net.seen += net.batch;  // 更新目前已经处理的图片数量: 每次处理一个batch, 故直接添加l.batch
    state.index = 0;  // 用于记录网络层编号
    state.net = net;  // 记录下当前的网络状态
    state.input = x;  // x中仍然包含batch张图片: batch * h * w * c
    state.delta = 0;  // 用于保存反向传播的梯度
    state.truth = y;  // ground_truth
    state.train = 1;  // 标记处于训练阶段
    forward_network(net, state);   // 前向传播
    backward_network(net, state);  // 反向传播
    float error = get_network_cost(net);
    //if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    if(*(state.net.total_bbox) > 0)
        fprintf(stderr, " total_bbox = %d, rewritten_bbox = %f %% \n",
                *(state.net.total_bbox), 100 * (float)*(state.net.rewritten_bbox) / *(state.net.total_bbox));
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;
    float* X = (float*)xcalloc(batch * d.X.cols, sizeof(float));
    float* y = (float*)xcalloc(batch * d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, X, y);
        net.current_subdivision = i;
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network(network net, data d)
{
    return train_network_waitkey(net, d, 0);
}


/*
** 训练一个batch(此处所指一个batch含有的图片是配置文件中真实指定的一个batch中含有的图片数量,
 * 也即图片张数为: net.batch*net.subdivision)
** 输入： net     已经构建好待训练的整个网络
**       d       此番训练所用到的所有图片数据(包含net.batch*net.subdivision张图片)
** 说明：train_network()函数对应处理ngpu=1的情况, 如果有多个gpu, 那么会首先调用train_networks()函数, 在其之中会调用get_data_part()
**      将在detector.c train_detector()函数中读入的net.batch*net.subdivision*ngpu张图片平分至每个gpu上, 
**      而后再调用train_network(), 总之最后train_work()的输入d中只包含net.batch*net.subdivision张图片
*/
float train_network_waitkey(network net, data d, int wait_key)
{
    // 事实上对于图像检测而言,d.X.rows/net.batch=net.subdivision,因此恒有d.X.rows % net.batch == 0,
    // 且下面的n就等于net.subdivision,(可以参看detector.c中的train_detector()),因此对于图像检测而言,
    // 下面三句略有冗余,但对于其他种情况(比如其他应用,非图像检测甚至非视觉情况),不知道是不是这样。
    assert(d.X.rows % net.batch == 0);
    // 注意: 这里net.batch=cfg.batch(也即配置中写的batch)/net.subdivisions
    // 因为在parse_network_cfg(cfgfile)时,有net.batch /= net.subvisions.
    int batch = net.batch;
    int n = d.X.rows / batch;  // n = net.subvisions
    float* X = (float*)xcalloc(batch * d.X.cols, sizeof(float));  // d.X.cols = h*w*c
    float* y = (float*)xcalloc(batch * d.y.cols, sizeof(float));
    int i;
    float sum = 0;
    /**
    * 因为darknet框架是将大batchsize分解成多个minibatch,实际每次forward()操作只有minibatch张图片.
    * 为了达到minibatch训练,batchsize反向传播效果,框架作者就是用误差累计策略.也即每次minibatch图片
    * forward()后,各层的误差项δ并不清零处理,而是进行累加操作.权重参数又需要根据δ来计算.通过n次forward()
    * 和backward()操作后,得到各层的累计梯度(比如: ∂L/∂w和∂L/∂b),最后在update_network(net)中统一更新权重,
    * 需要注意的是, 假设经过batchsize/minibatch次反向传播后, 我们得到了累计的梯度:
    * ∂L/∂w = 0(初始时刻) + ∂L_1/∂w + ∂L_2/∂w + ... + ∂L_n/∂w, 在update_network()函数中,
    * 会首先利用∂L/∂w进行权重权重更新: w_i+1 = w_i - lr * ∂L/∂w, 随后又对∂L/∂w乘以一个momentum项.
    * 在后续的更新中, 是继续在momentum*∂L/∂w的基础上继续进行梯度累加操作. 也即有:
    * ∂L/∂w = momentum*∂L/∂w + (∂L_1/∂w + ∂L_2/∂w + ... + ∂L_n/∂w). 可以发现, 其在update_network()
    * 函数中用于更新权重的∂L/∂w实际上是动量(指数滑动平均).
    * */
    for(i = 0; i < n; ++i){
        // 从d中读取batch张图片到net.input中,进行训练:
        // 第一个参数d包含了net.batch*net.subdivision张图片的数据,第二个参数batch即为每次循环
        // 读入到net.input也即参与train_network_datum(),训练的图片张数,第三个参数为在d中的偏移量,
        // 第四个参数为网络的输入数据,第五个参数为输入数据net.input对应的标签数据(gt).
        get_next_batch(d, batch, i*batch, X, y);
        net.current_subdivision = i;
        // 训练网络: 本次训练的数据共有net.batch张图片,
        // 训练包括一次前向过程: 计算每一层网络的输出并计算cost,
        // 一次反向过程: 计算误差项, 梯度、权重更新值、偏置更新值;
        float err = train_network_datum(net, X, y);  // X中仍然是包含batch(这个batch是被除net.subvisions的)张图片
        sum += err;
        if(wait_key) wait_key_cv(5);
    }
    (*net.cur_iteration) += 1;  // 每跑一个batch大小的数据, cur_iteration都会+1, net->seen则为: net->cur_iteration*batch*subdivs
#ifdef GPU
    update_network_gpu(net);
#else   // GPU
    update_network(net);  // 更新参数
#endif  // GPU
    free(X);
    free(y);
    return (float)sum/(n*batch);
}


float train_network_batch(network net, data d, int n)
{
    int i,j;
    network_state state={0};
    state.index = 0;
    state.net = net;
    state.train = 1;
    state.delta = 0;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = random_gen()%d.X.rows;
            state.input = d.X.vals[index];
            state.truth = d.y.vals[index];
            forward_network(net, state);
            backward_network(net, state);
            sum += get_network_cost(net);
        }
        update_network(net);
    }
    return (float)sum/(n*batch);
}

int recalculate_workspace_size(network *net)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    if (gpu_index >= 0) cuda_free(net->workspace);
#endif
    int i;
    size_t workspace_size = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        //printf(" %d: layer = %d,", i, l.type);
        if (l.type == CONVOLUTIONAL) {
            l.workspace_size = get_convolutional_workspace_size(l);
        }
        else if (l.type == CONNECTED) {
            l.workspace_size = get_connected_workspace_size(l);
        }
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        net->layers[i] = l;
    }

#ifdef GPU
    if (gpu_index >= 0) {
        printf("\n try to allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
        net->workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        printf(" CUDA allocate done! \n");
    }
    else {
        free(net->workspace);
        net->workspace = (float*)xcalloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;

#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i, cudnn_fastest, 0);
        }
        else if (net->layers[i].type == MAXPOOL) {
            cudnn_maxpool_setup(net->layers + i);
        }
#endif

    }
    recalculate_workspace_size(net); // recalculate workspace size
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    if(gpu_index >= 0){
        cuda_free(net->workspace);
        if (net->input_gpu) {
            cuda_free(*net->input_gpu);
            *net->input_gpu = 0;
            cuda_free(*net->truth_gpu);
            *net->truth_gpu = 0;
        }

        if (net->input_state_gpu) cuda_free(net->input_state_gpu);
        if (net->input_pinned_cpu) {
            if (net->input_pinned_cpu_flag) cudaFreeHost(net->input_pinned_cpu);
            else free(net->input_pinned_cpu);
        }
    }
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        //printf(" (resize %d: layer = %d) , ", i, l.type);
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }
        else if (l.type == CRNN) {
            resize_crnn_layer(&l, w, h);
        }else if (l.type == CONV_LSTM) {
            resize_conv_lstm_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if (l.type == LOCAL_AVGPOOL) {
            resize_maxpool_layer(&l, w, h);
        }else if (l.type == BATCHNORM) {
            resize_batchnorm_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if (l.type == YOLO) {
            resize_yolo_layer(&l, w, h);
        }else if (l.type == GAUSSIAN_YOLO) {
            resize_gaussian_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if (l.type == SHORTCUT) {
            resize_shortcut_layer(&l, w, h, net);
        }else if (l.type == SCALE_CHANNELS) {
            resize_scale_channels_layer(&l, net);
        }else if (l.type == SAM) {
            resize_sam_layer(&l, w, h);
        }else if (l.type == DROPOUT) {
            resize_dropout_layer(&l, inputs);
            l.out_w = l.w = w;
            l.out_h = l.h = h;
            l.output = net->layers[i - 1].output;
            l.delta = net->layers[i - 1].delta;
#ifdef GPU
            l.output_gpu = net->layers[i-1].output_gpu;
            l.delta_gpu = net->layers[i-1].delta_gpu;
#endif
        }else if (l.type == UPSAMPLE) {
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        } else if (l.type == REORG_OLD) {
            resize_reorg_old_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            fprintf(stderr, "Resizing type %d \n", (int)l.type);
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        inputs = l.outputs;
        net->layers[i] = l;
        //if(l.type != DROPOUT)
        {
            w = l.out_w;
            h = l.out_h;
        }
        //if(l.type == AVGPOOL) break;
    }
#ifdef GPU
    const int size = get_network_input_size(*net) * net->batch;
    if(gpu_index >= 0){
        printf(" try to allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
        net->workspace = cuda_make_array(0, workspace_size/sizeof(float) + 1);
        net->input_state_gpu = cuda_make_array(0, size);
        if (cudaSuccess == cudaHostAlloc(&net->input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
            net->input_pinned_cpu_flag = 1;
        else {
            cudaGetLastError(); // reset CUDA-error
            net->input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
            net->input_pinned_cpu_flag = 0;
        }
        printf(" CUDA allocate done! \n");
    }else {
        free(net->workspace);
        net->workspace = (float*)xcalloc(1, workspace_size);
        if(!net->input_pinned_cpu_flag)
            net->input_pinned_cpu = (float*)xrealloc(net->input_pinned_cpu, size * sizeof(float));
    }
#else
    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

detection_layer get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    detection_layer l = { (LAYER_TYPE)0 };
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

layer* get_network_layer(network* net, int i)
{
    return net->layers + i;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}

void top_predictions(network net, int k, int *index)
{
    int size = get_network_output_size(net);
    float *out = get_network_output(net);
    top_k(out, size, k, index);
}

// A version of network_predict that uses a pointer for the network
// struct to make the python binding work properly.
float *network_predict_ptr(network *net, float *input)
{
    return network_predict(*net, input);
}

float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state = {0};
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO) {
            s += yolo_num_detections(l, thresh);
        }
        if (l.type == GAUSSIAN_YOLO) {
            s += gaussian_yolo_num_detections(l, thresh);
        }
        if (l.type == DETECTION || l.type == REGION) {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

int num_detections_batch(network *net, float thresh, int batch)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO) {
            s += yolo_num_detections_batch(l, thresh, batch);
        }
        if (l.type == DETECTION || l.type == REGION) {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    int i;
    layer l = net->layers[net->n - 1];
    for (i = 0; i < net->n; ++i) {
        layer l_tmp = net->layers[i];
        if (l_tmp.type == YOLO || l_tmp.type == GAUSSIAN_YOLO || l_tmp.type == DETECTION || l_tmp.type == REGION) {
            l = l_tmp;
            break;
        }
    }

    int nboxes = num_detections(net, thresh);
    if (num) *num = nboxes;
    detection* dets = (detection*)xcalloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)xcalloc(l.classes, sizeof(float));
        // tx,ty,tw,th uncertainty
        if(l.type == GAUSSIAN_YOLO) dets[i].uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
        else dets[i].uc = NULL;

        if (l.coords > 4) dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
        else dets[i].mask = NULL;

        if(l.embedding_output) dets[i].embeddings = (float*)xcalloc(l.embedding_size, sizeof(float));
        else dets[i].embeddings = NULL;
        dets[i].embedding_size = l.embedding_size;
    }
    return dets;
}

detection *make_network_boxes_batch(network *net, float thresh, int *num, int batch)
{
    int i;
    layer l = net->layers[net->n - 1];
    for (i = 0; i < net->n; ++i) {
        layer l_tmp = net->layers[i];
        if (l_tmp.type == YOLO || l_tmp.type == GAUSSIAN_YOLO || l_tmp.type == DETECTION || l_tmp.type == REGION) {
            l = l_tmp;
            break;
        }
    }

    int nboxes = num_detections_batch(net, thresh, batch);
    assert(num != NULL);
    *num = nboxes;
    detection* dets = (detection*)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
        // tx,ty,tw,th uncertainty
        if (l.type == GAUSSIAN_YOLO) dets[i].uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
        else dets[i].uc = NULL;

        if (l.coords > 4) dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
        else dets[i].mask = NULL;

        if (l.embedding_output) dets[i].embeddings = (float*)xcalloc(l.embedding_size, sizeof(float));
        else dets[i].embeddings = NULL;
        dets[i].embedding_size = l.embedding_size;
    }
    return dets;
}

void custom_get_region_detections(layer l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
    box* boxes = (box*)xcalloc(l.w * l.h * l.n, sizeof(box));
    float** probs = (float**)xcalloc(l.w * l.h * l.n, sizeof(float*));
    int i, j;
    for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, map);
    for (j = 0; j < l.w*l.h*l.n; ++j) {
        dets[j].classes = l.classes;
        dets[j].bbox = boxes[j];
        dets[j].objectness = 1;
        for (i = 0; i < l.classes; ++i) {
            dets[j].prob[i] = probs[j][i];
        }
    }

    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    //correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
    correct_yolo_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int prev_classes = -1;
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == YOLO) {
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
            dets += count;
            if (prev_classes < 0) prev_classes = l.classes;
            else if (prev_classes != l.classes) {
                printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
                    prev_classes, l.classes);
            }
        }
        if (l.type == GAUSSIAN_YOLO) {
            int count = get_gaussian_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
            dets += count;
        }
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if (l.type == DETECTION) {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

void fill_network_boxes_batch(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter, int batch)
{
    int prev_classes = -1;
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == YOLO) {
            int count = get_yolo_detections_batch(l, w, h, net->w, net->h, thresh, map, relative, dets, letter, batch);
            dets += count;
            if (prev_classes < 0) prev_classes = l.classes;
            else if (prev_classes != l.classes) {
                printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
                    prev_classes, l.classes);
            }
        }
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if (l.type == DETECTION) {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].uc) free(dets[i].uc);
        if (dets[i].mask) free(dets[i].mask);
        if (dets[i].embeddings) free(dets[i].embeddings);
    }
    free(dets);
}

void free_batch_detections(det_num_pair *det_num_pairs, int n)
{
    int  i;
    for(i=0; i<n; ++i)
        free_detections(det_num_pairs[i].dets, det_num_pairs[i].num);
    free(det_num_pairs);
}

// JSON format:
//{
// "frame_id":8990,
// "objects":[
//  {"class_id":4, "name":"aeroplane", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.793070},
//  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.265497}
// ]
//},

char *detection_to_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, char *filename)
{
    const float thresh = 0.005; // function get_network_boxes() has already filtred dets by actual threshold

    char *send_buf = (char *)calloc(1024, sizeof(char));
    if (!send_buf) return 0;
    if (filename) {
        sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"filename\":\"%s\", \n \"objects\": [ \n", frame_id, filename);
    }
    else {
        sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"objects\": [ \n", frame_id);
    }

    int i, j;
    int class_id = -1;
    for (i = 0; i < nboxes; ++i) {
        for (j = 0; j < classes; ++j) {
            int show = strncmp(names[j], "dont_show", 9);
            if (dets[i].prob[j] > thresh && show)
            {
                if (class_id != -1) strcat(send_buf, ", \n");
                class_id = j;
                char *buf = (char *)calloc(2048, sizeof(char));
                if (!buf) return 0;
                //sprintf(buf, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f}",
                //    image_id, j, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]);

                sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, \"width\":%f, \"height\":%f}, \"confidence\":%f}",
                    j, names[j], dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]);

                int send_buf_len = strlen(send_buf);
                int buf_len = strlen(buf);
                int total_len = send_buf_len + buf_len + 100;
                send_buf = (char *)realloc(send_buf, total_len * sizeof(char));
                if (!send_buf) {
                    if (buf) free(buf);
                    return 0;// exit(-1);
                }
                strcat(send_buf, buf);
                free(buf);
            }
        }
    }
    strcat(send_buf, "\n ] \n}");
    return send_buf;
}


float *network_predict_image(network *net, image im)
{
    //image imr = letterbox_image(im, net->w, net->h);
    float *p;
    if(net->batch != 1) set_batch_network(net, 1);
    if (im.w == net->w && im.h == net->h) {
        // Input image is the same size as our net, predict on that image
        p = network_predict(*net, im.data);
    }
    else {
        // Need to resize image to the desired size for the net
        image imr = resize_image(im, net->w, net->h);
        p = network_predict(*net, imr.data);
        free_image(imr);
    }
    return p;
}

det_num_pair* network_predict_batch(network *net, image im, int batch_size, int w, int h, float thresh, float hier, int *map, int relative, int letter)
{
    network_predict(*net, im.data);
    det_num_pair *pdets = (struct det_num_pair *)calloc(batch_size, sizeof(det_num_pair));
    int num;
    int batch;
    for(batch=0; batch < batch_size; batch++){
        detection *dets = make_network_boxes_batch(net, thresh, &num, batch);
        fill_network_boxes_batch(net, w, h, thresh, hier, map, relative, dets, letter, batch);
        pdets[batch].num = num;
        pdets[batch].dets = dets;
    }
    return pdets;
}

float *network_predict_image_letterbox(network *net, image im)
{
    //image imr = letterbox_image(im, net->w, net->h);
    float *p;
    if (net->batch != 1) set_batch_network(net, 1);
    if (im.w == net->w && im.h == net->h) {
        // Input image is the same size as our net, predict on that image
        p = network_predict(*net, im.data);
    }
    else {
        // Need to resize image to the desired size for the net
        image imr = letterbox_image(im, net->w, net->h);
        p = network_predict(*net, imr.data);
        free_image(imr);
    }
    return p;
}

int network_width(network *net) { return net->w; }
int network_height(network *net) { return net->h; }

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float* X = (float*)xcalloc(net.batch * test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float* X = (float*)xcalloc(net.batch * test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;
}

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den);
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network_ptr(network* net)
{
    free_network(*net);
}

void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    free(net.layers);

    free(net.seq_scales);
    free(net.scales);
    free(net.steps);
    free(net.seen);
    free(net.cur_iteration);
    free(net.total_bbox);
    free(net.rewritten_bbox);

#ifdef GPU
    if (gpu_index >= 0) cuda_free(net.workspace);
    else free(net.workspace);
    free_pinned_memory();
    if (net.input_state_gpu) cuda_free(net.input_state_gpu);
    if (net.input_pinned_cpu) {   // CPU
        if (net.input_pinned_cpu_flag) cudaFreeHost(net.input_pinned_cpu);
        else free(net.input_pinned_cpu);
    }
    if (*net.input_gpu) cuda_free(*net.input_gpu);
    if (*net.truth_gpu) cuda_free(*net.truth_gpu);
    if (net.input_gpu) free(net.input_gpu);
    if (net.truth_gpu) free(net.truth_gpu);

    if (*net.input16_gpu) cuda_free(*net.input16_gpu);
    if (*net.output16_gpu) cuda_free(*net.output16_gpu);
    if (net.input16_gpu) free(net.input16_gpu);
    if (net.output16_gpu) free(net.output16_gpu);
    if (net.max_input16_size) free(net.max_input16_size);
    if (net.max_output16_size) free(net.max_output16_size);
#else
    free(net.workspace);
#endif
}

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

static float lrelu(float src) {
    const float eps = 0.001;
    if (src > eps) return src;
    return eps;
}

void fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->share_layer != NULL) {
                l->batch_normalize = 0;
            }

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    const size_t filter_size = l->size*l->size*l->c / l->groups;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] = (double)l->weights[w_index] * l->scales[f] / (sqrt((double)l->rolling_variance[f] + .00001));
                    }
                }

                free_convolutional_batchnorm(l);
                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else  if (l->type == SHORTCUT && l->weights && l->weights_normalization)
        {
            if (l->nweights > 0) {
                //cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
                int i;
                for (i = 0; i < l->nweights; ++i) printf(" w = %f,", l->weights[i]);
                printf(" l->nweights = %d, j = %d \n", l->nweights, j);
            }

            // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
            const int layer_step = l->nweights / (l->n + 1);    // 1 or l.c or (l.c * l.h * l.w)

            int chan, i;
            for (chan = 0; chan < layer_step; ++chan)
            {
                float sum = 1, max_val = -FLT_MAX;

                if (l->weights_normalization == SOFTMAX_NORMALIZATION) {
                    for (i = 0; i < (l->n + 1); ++i) {
                        int w_index = chan + i * layer_step;
                        float w = l->weights[w_index];
                        if (max_val < w) max_val = w;
                    }
                }

                const float eps = 0.0001;
                sum = eps;

                for (i = 0; i < (l->n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l->weights[w_index];
                    if (l->weights_normalization == RELU_NORMALIZATION) sum += lrelu(w);
                    else if (l->weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
                }

                for (i = 0; i < (l->n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l->weights[w_index];
                    if (l->weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
                    else if (l->weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;
                    l->weights[w_index] = w;
                }
            }

            l->weights_normalization = NO_NORMALIZATION;

#ifdef GPU
            if (gpu_index >= 0) {
                push_shortcut_layer(*l);
            }
#endif
        }
        else {
            //printf(" Fusion skip layer type: %d \n", l->type);
        }
    }
}

void forward_blank_layer(layer l, network_state state) {}

void calculate_binary_weights(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->xnor) {
                //printf("\n %d \n", j);
                //l->lda_align = 256; // 256bit for AVX2    // set in make_convolutional_layer()
                //if (l->size*l->size*l->c >= 2048) l->lda_align = 512;

                binary_align_weights(l);

                if (net.layers[j].use_bin_output) {
                    l->activation = LINEAR;
                }

#ifdef GPU
                // fuse conv_xnor + shortcut -> conv_xnor
                if ((j + 1) < net.n && net.layers[j].type == CONVOLUTIONAL) {
                    layer *sc = &net.layers[j + 1];
                    if (sc->type == SHORTCUT && sc->w == sc->out_w && sc->h == sc->out_h && sc->c == sc->out_c)
                    {
                        l->bin_conv_shortcut_in_gpu = net.layers[net.layers[j + 1].index].output_gpu;
                        l->bin_conv_shortcut_out_gpu = net.layers[j + 1].output_gpu;

                        net.layers[j + 1].type = BLANK;
                        net.layers[j + 1].forward_gpu = forward_blank_layer;
                    }
                }
#endif  // GPU
            }
        }
    }
    //printf("\n calculate_binary_weights Done! \n");

}

void copy_cudnn_descriptors(layer src, layer *dst)
{
#ifdef CUDNN
    dst->normTensorDesc = src.normTensorDesc;
    dst->normDstTensorDesc = src.normDstTensorDesc;
    dst->normDstTensorDescF16 = src.normDstTensorDescF16;

    dst->srcTensorDesc = src.srcTensorDesc;
    dst->dstTensorDesc = src.dstTensorDesc;

    dst->srcTensorDesc16 = src.srcTensorDesc16;
    dst->dstTensorDesc16 = src.dstTensorDesc16;
#endif // CUDNN
}

void copy_weights_net(network net_train, network *net_map)
{
    int k;
    for (k = 0; k < net_train.n; ++k) {
        layer *l = &(net_train.layers[k]);
        layer tmp_layer;
        copy_cudnn_descriptors(net_map->layers[k], &tmp_layer);
        net_map->layers[k] = net_train.layers[k];
        copy_cudnn_descriptors(tmp_layer, &net_map->layers[k]);

        if (l->type == CRNN) {
            layer tmp_input_layer, tmp_self_layer, tmp_output_layer;
            copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
            copy_cudnn_descriptors(*net_map->layers[k].self_layer, &tmp_self_layer);
            copy_cudnn_descriptors(*net_map->layers[k].output_layer, &tmp_output_layer);
            net_map->layers[k].input_layer = net_train.layers[k].input_layer;
            net_map->layers[k].self_layer = net_train.layers[k].self_layer;
            net_map->layers[k].output_layer = net_train.layers[k].output_layer;
            //net_map->layers[k].output_gpu = net_map->layers[k].output_layer->output_gpu;  // already copied out of if()

            copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
            copy_cudnn_descriptors(tmp_self_layer, net_map->layers[k].self_layer);
            copy_cudnn_descriptors(tmp_output_layer, net_map->layers[k].output_layer);
        }
        else if(l->input_layer) // for AntiAliasing
        {
            layer tmp_input_layer;
            copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
            net_map->layers[k].input_layer = net_train.layers[k].input_layer;
            copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
        }
        net_map->layers[k].batch = 1;
        net_map->layers[k].steps = 1;
    }
}


// combine Training and Validation networks
network combine_train_valid_networks(network net_train, network net_map)
{
    network net_combined = make_network(net_train.n);
    layer *old_layers = net_combined.layers;
    net_combined = net_train;
    net_combined.layers = old_layers;
    net_combined.batch = 1;

    int k;
    for (k = 0; k < net_train.n; ++k) {
        layer *l = &(net_train.layers[k]);
        net_combined.layers[k] = net_train.layers[k];
        net_combined.layers[k].batch = 1;

        if (l->type == CONVOLUTIONAL) {
#ifdef CUDNN
            net_combined.layers[k].normTensorDesc = net_map.layers[k].normTensorDesc;
            net_combined.layers[k].normDstTensorDesc = net_map.layers[k].normDstTensorDesc;
            net_combined.layers[k].normDstTensorDescF16 = net_map.layers[k].normDstTensorDescF16;

            net_combined.layers[k].srcTensorDesc = net_map.layers[k].srcTensorDesc;
            net_combined.layers[k].dstTensorDesc = net_map.layers[k].dstTensorDesc;

            net_combined.layers[k].srcTensorDesc16 = net_map.layers[k].srcTensorDesc16;
            net_combined.layers[k].dstTensorDesc16 = net_map.layers[k].dstTensorDesc16;
#endif // CUDNN
        }
    }
    return net_combined;
}

void free_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) free_state_conv_lstm(net.layers[k]);
        if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}

void randomize_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) randomize_state_conv_lstm(net.layers[k]);
        if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}


void remember_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) remember_state_conv_lstm(net.layers[k]);
        //if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}

void restore_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) restore_state_conv_lstm(net.layers[k]);
        if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}
