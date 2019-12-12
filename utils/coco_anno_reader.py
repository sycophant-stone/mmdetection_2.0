import os
import json
from pycocotools.coco import COCO


class CocoAnno(object):
    def __init__(self, src_coco_anno_file):
        if not os.path.exists(src_coco_anno_file):
            raise Exception("%s not exists!" % (src_coco_anno_file))
        self.coco = COCO(src_coco_anno_file)

    def get_img_id_list(self):
        return self.coco.getImgIds()

    def get_img_info_by_img_ids(self, img_ids):
        return self.coco.loadImgs(img_ids)

    def get_img_paths_by_img_ids(self, img_ids):
        imginfolist = self.coco.loadImgs(img_ids)
        id_path_img_map = {}
        for indx, imginfo in enumerate(imginfolist):
            imgpath = imginfo['file_name']
            id_path_img_map[indx] = imgpath

        return id_path_img_map

    def get_anno_ids(self, imgids, catids):
        return self.coco.getAnnIds(imgIds=imgids, catIds=catids, iscrowd=None)

    def get_anno_info_by_id(self, annoid):
        return self.coco.loadAnns(annoid)


def Test_gen_sub_datasets(src_imgid_list, src_ref_coco_file, dst_subset_coco_file):
    cocoanno = CocoAnno(src_coco_anno_file=src_ref_coco_file)
    image_section = []
    annotations = []
    anno_cnt = 0
    for id, imgid in enumerate(src_imgid_list):
        imginfo_list = cocoanno.get_img_info_by_img_ids(imgid)
        images_dict = {}
        images_dict['file_name'] = imginfo_list[0]['file_name']
        images_dict['height'] = imginfo_list[0]['height']
        images_dict['width'] = imginfo_list[0]['width']
        images_dict['id'] = id
        image_section.append(images_dict)

        anno_id_list = cocoanno.get_anno_ids(imgids=imgid, catids=1)
        for anno_id in anno_id_list:
            annoinfo = cocoanno.get_anno_info_by_id(anno_id)
            anno_cnt = anno_cnt + 1
            anno = {}
            anno['image_id'] = imgid
            anno['category_id'] = annoinfo[0]['category_id']
            anno['bbox'] = annoinfo[0]['bbox']
            anno['id'] = anno_cnt
            anno['area'] = annoinfo[0]['area']
            anno['iscrowd'] = annoinfo[0]['iscrowd']
            annotations.append(anno)

    categories = []
    cate = {}
    cate['supercategory'] = 'head'
    cate['name'] = 'head'
    cate['id'] = 1
    categories.append(cate)

    ann_js = {}
    ann_js['images'] = image_section
    ann_js['categories'] = categories
    ann_js['annotations'] = annotations
    json.dump(ann_js, open(dst_subset_coco_file, 'w'), indent=4)


if __name__ == '__main__':
    src_ref_coco_file = "/ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json"
    Test_gen_sub_datasets(
        src_imgid_list=[0, 1, 2],
        src_ref_coco_file=src_ref_coco_file,
        dst_subset_coco_file='dst_sub_train.json'
    )
