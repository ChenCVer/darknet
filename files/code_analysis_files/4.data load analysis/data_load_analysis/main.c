#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "src/utils.h"

// TODO: 关于C语言多线程的知识可以参考:https://www.cnblogs.com/zzdbullet/p/9526130.html

pthread_mutex_t mtx_load_data = PTHREAD_MUTEX_INITIALIZER;

void* load_thread(void* ptr){
    load_args a = *(load_args*)ptr;
    // 需要获取的样本数
    int sample_num = a.n;
    // 这里用随机数模拟一下
    int i, rand_data;
    // 该数据用于后期拼装
    a.d->rand_datas = (int*)malloc(a.n*sizeof(int));
    a.d->num_rand_datas = sample_num;

    for(i = 0; i < sample_num; i++){
        rand_data = rand() % 100;
        (a.d->rand_datas)[i] = rand_data;
    }

    pthread_mutex_lock(&mtx_load_data);    // 互斥锁上锁, 阻塞调用.

    printf("thread_id[%d]: rand_datas = ", a.thread_id);
    for(i = 0; i < sample_num; i++)
        printf(" %d ",a.d->rand_datas[i]);
    printf("\n");

    pthread_mutex_unlock(&mtx_load_data);  // 互斥锁解锁
}


pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    load_args *ptr = calloc(1, sizeof(load_args));
    *ptr = args;
    // 初始化随机种子
    srand((unsigned )time(NULL));
    if(pthread_create(&thread, 0, load_thread, ptr))
        error("Thread creation failed");
    return thread;
}


void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    // out和args.d指向的内存空间首地址一样, args.d指向buffer, out也就指向buffer
    data *out = args.d;
    int total = args.n;
    free(ptr);

    // 开辟args.threads个data类型大小的内存空间, buffers是data*类型
    // buffers指向:[data_0, data_1,...,data_args_threads-1]
    data* buffers = calloc(args.threads, sizeof(data));
    // 开辟args.threads个pthread_t类型大小的内存空间, threads是pthread_t*类型
    pthread_t* threads = calloc(args.threads, sizeof(pthread_t));

    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;  // args.d = buffers[i];
        // args.n也即buffers[i]->rand_datas装载的样本个数
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        // load_data_in_thread负责创建并启动子线程, 并返回线程ID给threads[i].
        args.thread_id = i;
        threads[i] = load_data_in_thread(args);  // 这里创建并启动子线程.
    }

    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);  // 等待所有子线程的数据load完毕.
    }

    // concat_datas()负责将buffers中的每个data中的rand_datas拼起来, 放在out所指向的内存空间.
    // 由之前的代码可以看出, out指向buffer数组, 这样最终通过concat_datas函数, 实现了将结果从函数
    // 传出到主函数main中.
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;  // shallow标记作用是什么?

    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }

    free(buffers);
    free(threads);

    return 0;
}


pthread_t load_data(load_args args)
{
    pthread_t thread;
    load_args *ptr = calloc(1, sizeof(load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr))
        error("Thread creation failed");
    return thread;
}


int main(int argc, char** argv) {
    int i;
    data train, buffer;
    load_args args = {0};
    args.n = 32;            // batchsize
    args.threads = 7;       // 线程数
    args.d = &buffer;       // args.d指向buffer

    pthread_t load_thread = load_data(args);
    pthread_join(load_thread, 0);  // 主程序等待load_thread线程执行完毕, 再往下执行
    train = buffer;
    printf("data load compeleted...\n");
    printf("-------------------------\n");
    printf("train.rand_datas = ");
    for(i = 0; i < args.n; i++)
        printf("%d ", train.rand_datas[i]);
    printf("\n");

    return 0;
}
