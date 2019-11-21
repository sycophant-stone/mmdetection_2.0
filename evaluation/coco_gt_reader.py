import os
import random
from utils.coco_anno_reader import CocoAnno
class GtReaderCoCo(object):
    def __init__(self,src_annotation_file):
        '''
        :param src_annotation_file:
        '''
        self.src_annoattion_file = src_annotation_file
        self.cocoanno = CocoAnno(src_coco_anno_file=src_annotation_file)
        self.imgid_path_map = {}
        self.imgid_bboxes_map = {}

    def gen_gt_img_bboxes_map(self):
        '''
        gen gt img bboxes map
        :return:
            imgid_path_map:
                id: img_base_name.ll
            imgid_bboxes_map:
                id:{
                    'xmin,ymin,xmax,ymax',
                    ...
                }
        '''
        cocoanno = self.cocoanno
        imgid_list = cocoanno.get_img_id_list()
        id_img_path_map = cocoanno.get_img_paths_by_img_ids(imgid_list)
        imgid_path_map={}
        imgid_bboxes_map={}
        for id,imgpath in id_img_path_map.items():
            anno_id_list = cocoanno.get_anno_ids(imgids=id, catids=1)
            anno_bboxes =[]
            if id not in imgid_bboxes_map:
                imgid_bboxes_map[id]=[] # have sequence inline.
            for anno_id in anno_id_list:
                annoinfo = cocoanno.get_anno_info_by_id(anno_id)
                bbox = annoinfo[0]['bbox']
                bbox = [bbox[0],bbox[1],bbox[0]+bbox[2], bbox[1]+bbox[3]]
                bbox = [str(b) for b in bbox]
                bbox.append("0") # 0 for ignore
                imgid_bboxes_map[id].append(",".join(bbox))

            imgid_path_map[id] = os.path.basename(imgpath)

        self.imgid_path_map = imgid_path_map
        self.imgid_bboxes_map = imgid_bboxes_map
        return imgid_path_map,imgid_bboxes_map

    def get_bbox_by_imgid_annoid(self, imgid, annoid):
        try:
            return self.imgid_bboxes_map[imgid][annoid]
        except:
            raise Exception("exception:   self.imgid_bboxes_map  with imgid:%s, annoid:%s"%(imgid, annoid))

    def get_annoidlist_for_specific_imgid(self,imgid):
        try:
            raw_anno_idlist = self.cocoanno.get_anno_ids(imgids=imgid, catids=1)
            # [28634, 28635]
            # and we wrap this to [0,1]
            return [int(b)-int(raw_anno_idlist[0]) for b in raw_anno_idlist]
        except:
            raise Exception("imgid:%s is invalid "%(imgid))




def Test_get_bbox_by_imgid_annoid():
    src_annotation_file = "/ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out/dataset/eva/source/instances_train.json"
    cgr = GtReaderCoCo(src_annotation_file)
    imgid_path_map,imgid_gt_bboxes_map = cgr.gen_gt_img_bboxes_map()
    annoid = 0
    imglist = range(0,len(imgid_path_map))
    imglist = random.sample(imglist, 5)
    for imgid in imglist:
        bboxes = cgr.get_bbox_by_imgid_annoid(imgid, annoid)
        print("imgid:%s,imgfile:%s, annoid:%s, bbox: %s"%(imgid, imgid_path_map[imgid], annoid, bboxes))


if __name__=='__main__':
    Test_get_bbox_by_imgid_annoid()