#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define  INT_MAX  65536

// ========================== 1. 相关数据结构的定义 ========================= //

// 链表上的节点
typedef struct node{
    void *val; //前节点的内容是一个void类型的空指针
    struct node *next; //指向当前节点的下一节点
    struct node *prev; //指向当前节点的上一节点
} node;


//双向链表: 保存所有网络结构参数
typedef struct list{
    int size; //list的所有节点个数
    node *front; //list的首节点
    node *back; //list的普通节点
} list;


// 定义section: 保存网络中每一层的网络类型和参数
typedef struct section{
    char *type;     // 网络类型
    list *options;  // 网络参数
}section;


// kvp 键值对(字典): 保存解析后的参数变量和参数值
typedef struct kvp{
    char *key;
    char *val;
    int used;
} kvp;


//1. 创建一个list, 取名sections, 记录一共有多少个section(一个section储存了某一层网络层所需参数)
//2. 然后创建一个node, 该node的void*类型指针用来指向新创建的section，而section的char*类型指针指
//   向.cfg文件中的某一行section中的list指针指向一个新创建的node，该node的void*指针指向一个kvp结
//   构体，kvp结构体中的key就是.cfg文件中的关键字(如batch, subdivisions等), val就是对应的值.

// ===============================2. 相关工具类函数================================== //

void malloc_error()
{
    fprintf(stderr, "xMalloc error\n");
    exit(EXIT_FAILURE);
}


void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if(!ptr) {
        malloc_error();
    }
    return ptr;
}


//初始化链表
list* make_list()
{
    list* l = (list*)xmalloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(EXIT_FAILURE);
}

void realloc_error()
{
    fprintf(stderr, "Realloc error\n");
    exit(EXIT_FAILURE);
}


void *xrealloc(void *ptr, size_t size) {
    ptr=realloc(ptr,size);
    if(!ptr)
    {
        realloc_error();
    }
    return ptr;
}


char* fgetl(FILE *fp)
{
    // feof()函数能判断任何类型的文件是否结尾(如果文件到了结尾，返回真)
    // Note: 如果第一次没有对文件进行读操作(也即fgetc()等操作), 直接调用此函数，则永远返回假(文件没有结束)
    if(feof(fp)){
        return 0;
    }
    size_t size = 512;
    char* line = (char*)xmalloc(size * sizeof(char));
    // 默认遇到换行符，结束读取，换行符都放在line中可以认为fgets()按行读取
    // 从fp所关联的文件中读取内容, 放到line中，一次最大读取为size个。
    if(!fgets(line, size, fp))
    {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp))
    {
        if(curr == size-1){
            size *= 2;
            line = (char*)xrealloc(line, size * sizeof(char));
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }

    // https://blog.csdn.net/wearlee/article/details/79769138
    // '\0'是一个字符,作为字符串结尾字符.它是一个字节大小, 占8位,0x00是16进制表示, 因为它以0x开头,
    // 0x00转换为2进制为00000000,正好八位, 在计算机内存中其实和'\0'的表示是一样的.
    if(curr >= 2)
        if(line[curr-2] == 0x0d)  //  0x0d <=> '\n'
            line[curr-2] = 0x00;

    if(curr >= 1)
        if(line[curr-1] == 0x0a)  // 0x0a <=> '\r'
            line[curr-1] = 0x00;

    return line;
}

/*
 * 简介: 将 val 指针插入 list 结构体 l 中，这里相当于是用 C 实现了 C++ 中的
 *         list 的元素插入功能
 * 参数: l    链表指针
 *         val  链表节点的元素值
 * 流程： list 中保存的是 node 指针. 因此，需要用 node 结构体将 val 包裹起来后才可以
 *       插入 list 指针 l 中
 * 注意: 此函数类似 C++ 的 insert() 插入方式；
 *      而 opion_insert() 函数类似 C++ map 的按值插入方式，比如 map[key]= value
 *      两个函数操作对象都是 list 变量， 只是操作方式略有不同。
*/
void list_insert(list *l, void *val)
{
    node* newnode = (node*)xmalloc(sizeof(node));
    newnode->val = val;
    newnode->next = 0;
    // 如果 list 的 back 成员为空(初始化为 0), 说明 l 到目前为止，还没有存入数据
    // 另外, 令 l 的 front 为 new （此后 front 将不会再变，除非删除）
    if(!l->back){
        l->front = newnode;
        newnode->prev = 0;
    }
    else{
        l->back->next = newnode;
        newnode->prev = l->back;
    }
    l->back = newnode;
    ++l->size;
}


void option_insert(list *l, char *key, char *val)
{
    kvp* p = (kvp*)xmalloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}


void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n'||c =='\r'||c==0x0d||c==0x0a)
            ++offset;
        else
            s[i-offset] = c;
    }
    s[len-offset] = '\0';
}


int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

// ===============================3. 核心函数================================== //

/*
 * 读取神经网络结构配置文件（.cfg文件）中的配置数据， 将每个神经网络层参数读取到每个
 * section 结构体 (每个 section 是 sections 的一个节点) 中， 而后全部插入到
 * list 结构体 sections 中并返回
 *
 * \param: filename    C 风格字符数组， 神经网络结构配置文件路径
 *
 * \return: list 结构体指针，包含从神经网络结构配置文件中读入的所有神经网络层的参数
 * 每个 section 的所在行的开头是 ‘[’ , ‘\0’ , ‘#’ 和 ‘;’ 符号开头的行为无效行, 除此
 *之外的行为 section 对应的参数行. 每一行都是一个等式, 类似键值对的形式.

 *可以看到, 如果某一行开头是符号 ‘[’ , 说明读到了一个新的 section: current, 然后第1508行
 *list_insert(options, current);` 将该新的 section 保存起来.

 *在读取到下一个开头符号为 ‘[’ 的行之前的所有行都是该 section 的参数, 在第 1518 行
 *read_option(line, current->options) 将读取到的参数保存在 current 变量的 options 中.
 *注意, 这里保存在 options 节点中的数据为 kvp 键值对类型.

 *当然对于 kvp 类型的参数, 需要先将每一行中对应的键和值(用 ‘=’ 分割) 分离出来, 然后再
 *构造一个 kvp 类型的变量作为节点元素的数据.
 */
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    //一个section表示配置文件中的一个字段，也就是网络结构中的一层
    //因此，一个section将读取并存储某一层的参数以及该层的type
    if(file == 0)
        file_error(filename);
    char *line = NULL;
    int nu = 0; //当前读取行号
    list* sections = make_list(); //sections包含所有的神经网络层参数, 初始化sections
    section *current = 0;//当前读取到某一层
    while((line=fgetl(file)) != 0)
    {
        ++ nu;
        strip(line); //去除读入行中含有的空格符
        switch(line[0])
        {
            // 以 '[' 开头的行是一个新的 section , 其内容是层的 type
            // 比如 [net], [maxpool], [convolutional] ...
            case '[':
                current = (section*)xmalloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0': //空行
            case '#': //注释
            case ';': //空行 
                free(line); // 对于上述三种情况直接释放内存即可
                break;
            default:
                // 剩下的才真正是网络结构的数据，调用 read_option() 函数读取
                // 返回 0 说明文件中的数据格式有问题，将会提示错误
                if(!read_option(line, current->options))
                {
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    //关闭文件
    fclose(file);

    return sections;
}


int main(int argc, char** argv){
    char* file_name = "/home/cxj/Desktop/project/darknet_read_cfg/yolov2.cfg";
    list* net = read_cfg(file_name);
    return 0;
}