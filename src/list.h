#ifndef LIST_H
#define LIST_H

// 用于分类权重随机采样
typedef struct{
    int class_num;
    int weight_sum;         // 用来记录权重和
    char** labels_weights;  // 用来记录权重列表
}weights;

// 用于分类权重随机采样
typedef struct{
    char *type;        // 用来记录类别名
    int size;          // 用来记录每一类图片数量
    char** filepaths;  // 用来记录每一类所有图片的路径
}sets;

// 链表上的节点
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

//双向链表: 用于保存所有的网络参数
typedef struct list{
    int size;
    node *front;
    node *back;
} list;

// C与C++混合编程时, 被extern "C"修饰的部分按照C语言方式编译和连接的.
// 如果代码不知道是被c调用还是c++调用时, 就需要用__cplusplus extern "C"
// 来修饰, __cplusplus是c++定义的宏, 如果是c++调用的话, extern "C"声明会
// 有效({code}, code部分用C编译器编译, 而不是用C++编译器). 如果时C调用的话,
// 那么, extern "C"声明无效.
// 参考: https://www.cnblogs.com/TurboLemon/p/6364241.html
#ifdef __cplusplus
extern "C" {
#endif
list *make_list();  // 初始化链表
int list_find(list *l, void *val);

void list_insert(list *, void *);

void **list_to_array(list *l);

void free_list_val(list *l);
void free_list(list *l);
void free_list_contents(list *l);
void free_list_contents_kvp(list *l);

#ifdef __cplusplus
}
#endif
#endif
