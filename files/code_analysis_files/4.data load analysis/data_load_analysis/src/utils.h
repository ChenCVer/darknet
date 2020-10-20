#ifndef _UTILS_H
#define _UTILS_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

typedef struct matrix{
    int rows;
    int cols;
    float **vals;
}matrix;


typedef struct box {
    float x, y, w, h;
}box;


typedef struct data{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
    // 下面是演示用:
    int* rand_datas;
    int num_rand_datas;
}data;

typedef struct load_args{
    int threads;    // 线程数
    int thread_id;  // 记录开辟的第几个线程
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    data *d;
    int shallow;
}load_args;

void error(const char *s);

matrix concat_matrix(matrix m1, matrix m2);

data concat_data(data d1, data d2);

void free_matrix(matrix m);

void free_data(data d);

int* concat_rand_data(data rand_datas_1, data rand_datas_2);

data concat_datas(data *d, int n);

#endif
