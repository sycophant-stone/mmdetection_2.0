import os
def get_box_from_str_format(str_box):
    '''
    format then get boxes.
    :param str_box:
    :return:
    '''
    b = str_box.split(',')
    b = [float(bi) for bi in b]
    return b[0:5]


def listdata_saveto_csv_data(src_csv_data_list, dst_csv_file):
    '''
    save list [1,2,3...] to csv file.whose format is 1,2,3,...
    :param src_csv_data_list:
    :param dst_csv_file:
    :return:
    '''
    src_csv_data_list = [str(b) for b in src_csv_data_list]
    with open(dst_csv_file,'w') as f:
        for data in src_csv_data_list:
            f.write("%s\n"%(data))


def csv_file_to_listdata(src_csv_data_file):
    print(src_csv_data_file)
    if not os.path.exists(src_csv_data_file):
        print("%s not exists!"%(src_csv_data_file))

    src_csv_data_list=[]
    with open(src_csv_data_file,'r') as f:
        for line in f.readlines():
            src_csv_data_list.append(int(line.strip()))

        return src_csv_data_list

def csv_file_to_listdata_multilines(src_csv_data_file):
    print(src_csv_data_file)
    if not os.path.exists(src_csv_data_file):
        print("%s not exists!"%(src_csv_data_file))

    src_csv_data_list=[]
    with open(src_csv_data_file,'r') as f:
        for line in f.readlines():
            src_csv_data_list.append(int(line.strip()))

        return src_csv_data_list


def Test_csv_file_to_listdata():
    # src_csv_data_file = "/ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out/dataset/eva/result/fpfn_bad_cases.csv"
    # csv_file_to_listdata(src_csv_data_file)
    src_csv_data_file = "/ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out/dataset/eva/result/fpfn_bad_cases_one.csv"
    csv_file_to_listdata_multilines(src_csv_data_file)

if __name__=='__main__':
    Test_csv_file_to_listdata()