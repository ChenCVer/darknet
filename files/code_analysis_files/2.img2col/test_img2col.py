def get_pixel_value(data_im,
                    height,
                    width,
                    channels,
                    row,
                    col,
                    channel,
                    pad):
    row -= pad
    col -= pad

    if (row < 0 or col < 0 or
            row >= height or col >= width):
        return 0
    print("--------------------------------------------------------")
    index = row * height * channel + width * row + col
    print("row(im_row-1)=", row, ", col(im_col-1)=", col, ", index=", index)
    print("--------------------------------------------------------")

    return data_im[index]


def im2col(data_im,
           channels,
           height,
           width,
           ksize,
           stride,
           pad,
           data_col):

    height_col = int((height + 2 * pad - ksize) / stride + 1)  # 输出的高
    width_col = int((width + 2 * pad - ksize) / stride + 1)    # 输出的宽
    channels_col = channels * ksize * ksize

    for c in range(channels_col):
        # 当c=0时, 最终代码得到的填充的元素, 都位于data_col的第一行的位置(从二维数组的角度来说)
        # 从之后的矩阵运行来看, 假设卷积核为:A = [1 1 1 1 1 1 1 1 1], 重排的图像数据为B,
        # B的第一行永远与卷积核A的第0个元素进行相乘.

        # 从以上分析我们就可以得知, 当c=0, w_offset=0, h_offset=0, 在下面的两个for循环中,主要任务
        # 就是在data_im找出所有与卷积核的第[0, 0, 0]位置对应相乘的元素的位置, 并将其对应的值通过
        # get_pixel_value获取后填入到data_col[col_index]中, 说白了就是填写在data_col的第一行位置.
        # 当c=1, w_offset=1, h_offset=0, 在下面的两个for循环中,主要任务就是在data_im找出所有与卷积
        # 核的第[0, 1, 0]位置对应相乘的元素的位置, 并将其对应的值通过get_pixel_value获取后填入到
        # data_col[col_index]中, 说白了就是填写在data_col的第二行位置.
        # 当c=8, w_offset=2, h_offset=2, 在下面的两个for循环中,主要任务就是在data_im找出所有与卷积
        # 核的第[2, 2, 0]位置对应相乘的元素的位置, 并将其对应的值通过get_pixel_value获取后填入到
        # data_col[col_index]中, 说白了就是填写在data_col的第八行位置.
        w_offset = c % ksize             # 列偏移量
        h_offset = (c // ksize) % ksize  # 行偏移量
        c_im = c // ksize // ksize       # 通道偏移量
        print("----------------------(c = {0})----------------------".format(c))
        for h in range(height_col):
            for w in range(width_col):
                # h, w是输出特征图的位置索引, im_row, im_col是对应的在data_im(输入特征图)中的索引index.
                # 这个index是与卷积核的[h_offset, w_offset, c_im]位置对应.
                # 举个例子, 假设h_offset=1, w_offset=2, 则h=0, w=0, 则对应的就是卷积核在data_im上进行
                # 第一次卷积操作时, 卷积核的第[h_offset, w_offset, c_im]位置对应的data_im上的位置index,
                # 然后将data_im[index]填入到data_col[col_index]中.
                im_row = h_offset + h * stride  # 根据输出特征图的索引h, 计算对应data_im中的索引im_row
                im_col = w_offset + w * stride  # 根据输出特征图的索引w, 计算对应data_im中的索引im_col
                # [col_index % (height_col*width_col)] + 1 实质上可以认为是卷积核在data_im上第几次卷积.
                col_index = c * height_col * width_col + h * width_col + w
                print("h=", h, ", w=", w, ", im_row=", im_row, ", im_col=", im_col, ", col_index=", col_index)

                data_col[col_index] = get_pixel_value(data_im, height, width, channels,
                                                      im_row, im_col, c_im, pad)

    return data_col


if __name__ == '__main__':
    data_im = [1, 2, 3, 4, 5,
               6, 7, 8, 9, 10,
               11, 12, 13, 14, 15,
               16, 17, 18, 19, 20,
               21, 22, 23, 24, 25]
    pad = 1
    stride = 2
    ksize = 3
    height = 5
    width = 5
    chanels = 1
    data_col = 81 * [0, ]
    data_col = im2col(data_im, chanels, height, width,
                      ksize, stride, pad, data_col)

    for i in range(9):
        for j in range(9):
            print("{0}\t".format(data_col[i*9 + j]), end=' ')
        print()
