import os
import cv2
import mmcv
import argparse
import numpy as np
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector, show_result
import mmdet.ops.nms as mm_nms
from eval_fppi import *

#  test a list of images and write the results to image files
# mgs = ['test1.jpg', 'test2.jpg']
# or i, result in enumerate(inference_detector(model, imgs):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

def GET_FILE_BARENAME(filename):
    if not os.path.exists(filename):
        raise Exception("%s not exists"%(filename))
    return os.path.splitext(os.path.basename(filename))[0]


def model_init(src_config_file, src_checkpoint_file):
    '''
    model init by config file and checkpoint file
    :param src_config_file:
    :param src_checkpoint_file:
    :return: model
    '''
    model = init_detector(src_config_file, src_checkpoint_file, device='cuda:0')
    return model


def test_and_save(src_config_file, src_checkpoint_file, src_picture_file,input_show_info):
    '''
    test pictures and save them to jpgs.
    :param src_config_file:
    :param src_checkpoint_file:
    :param src_picture_file:
    :param input_show_info
    :return:
    '''
    # build the model from a config file and a checkpoint file
    print("src_config_file:", src_config_file)
    print("src_checkpoint_file:", src_checkpoint_file)
    print("src_picture_file:", src_picture_file)

    model = init_detector(src_config_file, src_checkpoint_file, device='cuda:0')

    # test a single image and show the results
    result = inference_detector(model, src_picture_file)
    class_names = model.CLASSES

    img = mmcv.imread(src_picture_file)
    cvimg = cv2.imread(src_picture_file)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    # print("bboxes:",bboxes)

    fined_boxes = mm_nms(bboxes, 0.2)
    # print(fined_boxes)
    # raise Exception("stop")
    for b in fined_boxes[0]:
        xmin,ymin,xmax,ymax = b[0:4]
        conf = float(b[4])
        if input_show_info:
            print(b)
        if conf > 0.1:
            cvimg = cv2.rectangle(cvimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
    cv2.imwrite(GET_FILE_BARENAME(src_picture_file)+'_out.jpg',cvimg)

def test_iteration_with_model(src_model, src_iter_imgpath, src_pred_dict, src_bool_gen_predbbox, dst_img_predbbox):
    '''
    test one img with given model and given imgpath.
    :param src_model:
    :param src_iter_imgpath:
    :param src_pred_dict: pred dict for saving predtions
    :param src_bool_gen_predbbox: bool flag, wether gen predbbox or not.
    :return:
    '''

    result = inference_detector(model, src_iter_imgpath)
    class_names = model.CLASSES

    img = mmcv.imread(src_iter_imgpath)
    cvimg = cv2.imread(src_iter_imgpath)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    # print("bboxes:",bboxes)

    fined_boxes = mm_nms(bboxes, 0.2)
    # print(fined_boxes)
    # raise Exception("stop")
    predlist = []
    for b in fined_boxes[0]:
        xmin,ymin,xmax,ymax = b[0:4]
        conf = float(b[4])
        if input_show_info:
            print(b)
        if conf > 0.1:
            cvimg = cv2.rectangle(cvimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
            predlist.append(b)
    src_pred_dict[src_iter_imgpath] = predlist

    if src_bool_gen_predbbox:
        outpath = dst_img_predbbox+'/'+GET_FILE_BARENAME(src_iter_imgpath)+'_out.jpg'
        cv2.imwrite(outpath,cvimg)





def read_coco_set(src_set_dir):
    '''
    get coco set's img and annotations.
    :param src_set_dir: coco set dir path
    :return: a dict for imgpath and annotations
             i) annotation's format is '91,14,117,117'
             ii) like:
                key:   '/ssd/hnren/Data/coco_300px_large_labelbox_head/FID_DID_HEAD_CLEAN_0_patches_int/annotations/../JPEGImages/ch01010_20190307_ch01010_20190307100742.mp4.cut.mp4_006000_crop_3.jpg':
                vau:   {'0,218,86,81', '108,108,83,83', '268,6,31,59', '229,70,67,67'}
            a dict for evaluation
            gt_result: a dict {label0:{file1:[[xmin, ymin, xmax, ymax, ignore], ...], ...}, ...}
    '''
    # path string preprocess.
    set_dirname = os.path.dirname(src_set_dir)
    imgpath_bbox_map = {}
    gt_result = {}
    head_gt_result = {}

    coco = COCO(src_set_dir)
    imgids = coco.getImgIds() # will return all the imgids of this very set.
    for imgid in imgids:
        imginfo_this_id = coco.loadImgs(imgid) # this will return the imginfo for this imgid, including filename(path). img width,height.
        filename = imginfo_this_id[0]['file_name']
        img_abs_path = set_dirname+"/"+filename

        if not os.path.exists(img_abs_path):
            print("%s not exits"%(img_abs_path))

        anno_ids_this_img = coco.getAnnIds(imgIds=imgid, catIds=[1], iscrowd=None)
        all_bboxes_list=[]
        for annoid in anno_ids_this_img:
            img_anno = coco.loadAnns(annoid)
            # print("img:%s"%(img_abs_path))
            # print("img_anno:%s"%(img_anno))
            bbox = img_anno[0]['bbox']

            if img_abs_path not in imgpath_bbox_map:
                imgpath_bbox_map[img_abs_path] = set([])
            imgpath_bbox_map[img_abs_path].add("{},{},{},{}".format(bbox[0], bbox[1],bbox[2],bbox[3]))

            tmp_bbox_list = bbox
            tmp_bbox_list.append(0) # add ignore
            all_bboxes_list.append(tmp_bbox_list)
        head_gt_result[img_abs_path]=all_bboxes_list

    gt_result['head'] = head_gt_result
    return imgpath_bbox_map,gt_result

'''
this script will test one picture, and draw those detection's predictions bounding boxes on the picture.
'''
if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--src_config_file', type=str, help='model config file path', required=True)
    parse.add_argument('--src_checkpoint_file', type=str, help='model checkpoint file path', required=True)
    parse.add_argument('--src_picture_file', type=str, help='target pictures path', required=True)
    parse.add_argument('--src_bool_gen_predbbox', type=bool, help='wether gen prebbox for version', required=True)
    parse.add_argument('--dst_img_predbbox', type=str, help='for saving predition results pictures', required=True)
    parse.add_argument('--show_info', type=bool, help='whether show more info', required=False,default=False)

    _args = parse.parse_args()
    input_src_config_file       = _args.src_config_file
    input_src_checkpoint_file   = _args.src_checkpoint_file
    input_src_picture_file      = _args.src_picture_file
    input_dst_img_predbbox      = _args.dst_img_predbbox
    input_src_bool_gen_predbbox = _args.src_bool_gen_predbbox
    input_show_info             = _args.show_info

    if not os.path.exists(input_dst_img_predbbox):
        os.mkdir(input_dst_img_predbbox)

    imgpath_bbox_map, gt_result = read_coco_set(src_set_dir=input_src_picture_file)
    # dt_result: a dict {label0:{file1:[[xmin, ymin, xmax, ymax, score], ...], ...}, ...}
    model = model_init(src_config_file=input_src_config_file, src_checkpoint_file=input_src_checkpoint_file)
    pred_result={}
    pred_dict={}
    num_imgs = len(imgpath_bbox_map.keys())
    print("num_imgs:%d"%(num_imgs))
    for index, iter_img_path in enumerate(imgpath_bbox_map.keys()):
        if index %(1000)==0:
            print("cur at %d/%d"%(index%(1000), num_imgs/1000))
        test_iteration_with_model(src_model=model,
                                  src_iter_imgpath=iter_img_path,
                                  src_pred_dict=pred_dict,
                                  src_bool_gen_predbbox=input_src_bool_gen_predbbox,
                                  dst_img_predbbox=input_dst_img_predbbox)

    pred_result['head']=pred_dict
    match = detection_result_match(gt_result, pred_result, 0.5)
    print(match)
