import numpy as np

def gen_data(src_array_size, dst_np_array):
    dst_np_array = np.random.randn(int(src_array_size))
    print("dst_np_array: ", dst_np_array)


def show_data(src_np_array):
    print("type of src_np_array: ", type(src_np_array))
    print("src_np_array: ", src_np_array)

