#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include "src/utils.h"


void* load_thread(void* ptr){
    load_args a = *(load_args*)ptr;
    // 需要获取的样本数
    int sample_num = a.n;
    // 这里用随机数模拟一下
    int i, rand_data;
    // 该数据用于后期拼装
    a.d->rand_datas = (int*)malloc(a.n*sizeof(int));

    for(i = 0; i < sample_num; i++){
        rand_data = rand() % 100;
        (a.d->rand_datas)[i] = rand_data;
    }

    printf("rand_datas = ");
    for(i = 0; i < sample_num; i++)
        printf(" %d ",a.d->rand_datas[i]);

    printf("\n");
}


pthread_t load_data_in_thread(struct load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr))
        error("Thread creation failed");
    return thread;
}


void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);

    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));

    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);  // 这里创建并启动子线程.
    }

    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);  // 等待所有子线程的数据load完毕.
    }

    // 将每个线程load的数据拼接在一起.用模拟数据rand_datas进行仿真
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;

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

    FILE* fp = NULL;
    char* filepath = "/home/cxj/Desktop/project/data_load_analysis/2007_train.txt";
    fp = fopen(filepath, "r");
    if(fp == NULL){
        perror("fopen");
        return -1;
    }
    char buf[100] = {0};
    int total_imgs = 0;
    while(!feof(fp)){
        char* p = fgets(buf, sizeof(buf), fp);
        if (p != NULL){
            total_imgs++;
        }
    }
    fclose(fp);
    fp = NULL;

    // 构建空间
    char** paths = (char**)malloc(total_imgs * sizeof(char*));
    int i = 0;
    FILE* fp2 = fopen(filepath, "r");
    if (fp2 == NULL){
        perror("fopen");
        return -1;
    }
    while (!feof(fp2)){
        char* p = fgets(buf, sizeof(buf), fp2);
        if(p != NULL) {
            paths[i] = malloc(100 * sizeof(char));
            memset(paths[i], 0, 100);
            strcpy(paths[i], p);
            i++;
        }
    }
    fclose(fp2);
    fp2 = NULL;

    data train, buffer;
    load_args args = {0};
    args.paths = paths;   // 训练数据路径的字符串数组
    args.m = total_imgs;  // 训练数据总样本数
    args.n = 16;          // batchsize
    args.threads = 3;     // 线程数
    args.d = &buffer;     // 获取数据的地址.

    pthread_t load_thread = load_data(args);
    pthread_join(load_thread, 0);
    printf("data load compeleted...\n");

    return 0;
}
