import json
import random

class PredReader(object):
    def __init__(self, src_pred_file):
        self.src_pred_file = src_pred_file
        self.sorted_ib_bboxes_map = {}

    def gen_pred_img_maps(self):
        '''
        pred file is json file
            [{"image_id": 0, "bbox": [87.8564453125, 92.02633666992188, 99.71109008789062, 98.91604614257812], "score": 0.24198056757450104, "category_id": 1},
            {"image_id": 0, "bbox": [0.0, 26.637065887451172, 109.99320220947266, 104.29878616333008], "score": 0.10887424647808075, "category_id": 1},
            ...

        :param src_pred_file:
            ...
            imgid:
                {
                    'xmin,ymin,xmax,ymax,score',    # bbox0
                    'xmin,ymin,xmax,ymax,score',    # bbox1
                    'xmin,ymin,xmax,ymax,score',    # bbox2
                }
            662:
                {
                    '251.01080322265625,7.740741729736328,46.945587158203125,94.00860214233398,0.04928673058748245',
                    '203.4139862060547,0.8575859069824219,94.72004699707031,102.16753005981445,0.06794986873865128',
                    '243.6862335205078,241.088623046875,56.13917541503906,58.911376953125,0.25228849053382874',
                    '24.294723510742188,172.23358154296875,63.590675354003906,86.3138427734375,0.019884994253516197',
                    '199.16452026367188,50.36572265625,75.95703125,82.05302429199219,0.02894858457148075',
                    '0  .0,0.0,41.14662170410156,40.11796188354492,0.02012813650071621',
                    ...
                }
        :return:
        '''
        src_pred_file = self.src_pred_file
        ib_bboxes_map={}
        with open(src_pred_file,'r') as f:
            ibs_es= json.load(f) # means that img, bboxes, score
            # print(type(result))
            for ibs in ibs_es:
                imgid = ibs['image_id']
                score = ibs['score']
                bbox = ibs['bbox']
                bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                bbox = [str(b) for b in bbox]
                bbox.append(str(score))

                ib_key = imgid
                if ib_key not in ib_bboxes_map:
                    ib_bboxes_map[ib_key]=set([])
                ib_bboxes_map[ib_key].add(",".join(bbox))

        sorted_ib_bboxes_map = {}
        for imgid, bboxes_string_set in ib_bboxes_map.items():
            bboxes_map = {}
            for bstring in bboxes_string_set:
                words = bstring.strip().split(',')
                score = float(words[-1])
                bbox = bstring# ",".join(words[:-1])
                bboxes_map[score] = bbox
            sorted_boxes_list = sorted(bboxes_map)
            # print("bboxes_map: ",bboxes_map)
            # print("sorted_boxes_list :",sorted_boxes_list)
            # sort by pred's bboxes' score..
            for sorted_cursor in sorted_boxes_list:
                if imgid not in sorted_ib_bboxes_map:
                    sorted_ib_bboxes_map[imgid] = []#set([])
                sorted_ib_bboxes_map[imgid].append(bboxes_map[sorted_cursor])
            # print("sorted_ib_bboxes_map:  ",sorted_ib_bboxes_map)

        self.sorted_ib_bboxes_map = sorted_ib_bboxes_map
        return sorted_ib_bboxes_map
        # return ib_bboxes_map

    def get_bboxes_by_imgid(self, imgid):
        '''
        get all the prediction bboxes from imgid.
        :param imgid:
        :return:
        '''
        try:
            return self.sorted_ib_bboxes_map[imgid]
        except:
            raise Exception("exception:  sorted_ib_bboxes_map  with imgid: %s "%(imgid))


def Test_get_bboxes_by_imgid():
    src_pred_file = "/ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out/dataset/eva/source/eva_r18.pkl.json"
    pr = PredReader(src_pred_file)
    imgid_pred_bboxes_map = pr.gen_pred_img_maps()
    imglist = range(0, len(imgid_pred_bboxes_map.keys()))
    imglist = random.sample(imglist, 5)
    for imgid in imglist:
        bboxes = pr.get_bboxes_by_imgid(imgid)
        print("imgid: %s  with bboxes:%s "%(imgid, bboxes))
    # then check on `eva_r18.pkl.json` with those output log.

if __name__=='__main__':
    Test_get_bboxes_by_imgid()