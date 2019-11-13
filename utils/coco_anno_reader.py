import os
import numpy as np
from pycocotools.coco import COCO

class CocoAnno(object):
    def __init__(self, src_coco_anno_file):
        if not os.path.exists(src_coco_anno_file):
            raise Exception("%s not exists!"%(src_coco_anno_file))
        self.coco = COCO(src_coco_anno_file)

    def get_img_id_list(self):
        return self.coco.getImgIds()
    def get_img_info_by_img_ids(self, img_ids):
        return self.coco.loadImgs(img_ids)

    def get_img_paths_by_img_ids(self,img_ids):
        imginfolist = self.coco.loadImgs(img_ids)
        id_path_img_map = {}
        for indx, imginfo in enumerate(imginfolist):
            imgpath = imginfo['file_name']
            id_path_img_map[indx] = imgpath

        return id_path_img_map
    def get_anno_ids(self,imgids,catids):
        return self.coco.getAnnIds(imgIds=imgids, catIds=catids, iscrowd=None)

    def get_anno_info_by_id(self, annoid):
        return self.coco.loadAnns(annoid)



if __name__=='__main__':
    cocoanno = CocoAnno(src_coco_anno_file="/ssd/hnren/Data/coco_300px_large_labelbox_head/FID_DID_HEAD_CLEAN_0_patches_int/annotations/instances_train.json")
    # print(cocoanno.get_img_id_list())
    imgid_list = cocoanno.get_img_id_list()
    imginfo_list = cocoanno.get_img_info_by_img_ids(imgid_list)
    # print(type(imginfo_list))
    id_img_path_map = cocoanno.get_img_paths_by_img_ids(imgid_list)
    # print(id_img_path_map)
    img_bboxes_map = {}
    for id,imgpath in id_img_path_map.items():
        anno_id_list = cocoanno.get_anno_ids(imgids=id, catids=1)
        anno_bboxes =[]
        for anno_id in anno_id_list:
            annoinfo = cocoanno.get_anno_info_by_id(anno_id)
            bbox = annoinfo[0]['bbox']
            anno_bboxes.append(bbox)

            # print(annoinfo)
        imgpath_key = os.path.basename(imgpath)
        img_bboxes_map[imgpath_key]=anno_bboxes

    print(img_bboxes_map)

