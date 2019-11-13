import os
import json
from utils.coco_anno_reader import CocoAnno

def gen_gt_img_bboxes_map(src_annotation_file):
    cocoanno = CocoAnno(src_coco_anno_file=src_annotation_file)
    imgid_list = cocoanno.get_img_id_list()
    id_img_path_map = cocoanno.get_img_paths_by_img_ids(imgid_list)
    img_bboxes_map = {}
    for id,imgpath in id_img_path_map.items():
        anno_id_list = cocoanno.get_anno_ids(imgids=id, catids=1)
        anno_bboxes =[]
        for anno_id in anno_id_list:
            annoinfo = cocoanno.get_anno_info_by_id(anno_id)
            bbox = annoinfo[0]['bbox']
            anno_bboxes.append(bbox)
        imgpath_key = os.path.basename(imgpath)
        img_bboxes_map[imgpath_key]=anno_bboxes

    return img_bboxes_map

def gen_pred_img_maps(src_pred_file):
    '''
    pred file is json file
        [{"image_id": 0, "bbox": [87.8564453125, 92.02633666992188, 99.71109008789062, 98.91604614257812], "score": 0.24198056757450104, "category_id": 1},
        {"image_id": 0, "bbox": [0.0, 26.637065887451172, 109.99320220947266, 104.29878616333008], "score": 0.10887424647808075, "category_id": 1},
        ...

    :param src_pred_file:
    :return:
    '''
    with open(src_pred_file,'r') as f:
        result = json.load(f)
        print(result)
    

def visionalize_cases(src_annotation_file, src_pred_file, dst_img_save):
    '''
    visionalization for cases.
    :param src_annotation_file:
    :param src_pred_file:
    :param dst_img_save:
    :return:
    '''
    img_bboxes_map = gen_gt_img_bboxes_map(src_annotation_file)
    gen_pred_img_maps(src_pred_file)
