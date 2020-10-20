#include "utils.h"


void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}


int* concat_rand_data(data data_1, data data_2){

    int i, count = 0;
    int num_rand_datas = (data_1.num_rand_datas+data_2.num_rand_datas);
    int* new_data = (int*)malloc(num_rand_datas * sizeof(int));

    for(i = 0; i < data_1.num_rand_datas; i++){
        new_data[count++] = data_1.rand_datas[i];
    }

    for(i = 0; i < data_2.num_rand_datas; i++){
        new_data[count++] = data_2.rand_datas[i];
    }

    return new_data;
}


matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}


data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.rand_datas = concat_rand_data(d1, d2);
    d.w = d1.w;
    d.h = d1.h;
    d.num_rand_datas = d1.num_rand_datas + d2.num_rand_datas;
    return d;
}


void free_matrix(matrix m)
{
    int i;
    for(i = 0; i < m.rows; ++i) free(m.vals[i]);
    free(m.vals);
}


void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}


data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){  // n个线程.
        // 把d[i]中的rand_datas数据提取出来放在out中.
        data new = concat_data(d[i], out);  // d[i]也即buffers[i]
        free_data(out);
        out = new;
    }
    return out;
}

