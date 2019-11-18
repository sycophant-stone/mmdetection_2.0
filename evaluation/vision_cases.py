import os
import cv2
import json
import evaluation.draw as draw
from utils.coco_anno_reader import CocoAnno
import evaluation.matrix as matrix

def gen_gt_img_bboxes_map(src_annotation_file):
    '''
    gen gt img bboxes map
    :param src_annotation_file:
    :return:
        imgid_path_map:
            id: img_base_name.ll
        imgid_bboxes_map:
            id:{
                'xmin,ymin,xmax,ymax',
                ...
            }
    '''
    cocoanno = CocoAnno(src_coco_anno_file=src_annotation_file)
    imgid_list = cocoanno.get_img_id_list()
    id_img_path_map = cocoanno.get_img_paths_by_img_ids(imgid_list)
    imgid_path_map={}
    imgid_bboxes_map={}
    for id,imgpath in id_img_path_map.items():
        anno_id_list = cocoanno.get_anno_ids(imgids=id, catids=1)
        anno_bboxes =[]
        if id not in imgid_bboxes_map:
            imgid_bboxes_map[id]=set([])
        for anno_id in anno_id_list:
            annoinfo = cocoanno.get_anno_info_by_id(anno_id)
            bbox = annoinfo[0]['bbox']
            bbox = [bbox[0],bbox[1],bbox[0]+bbox[2], bbox[1]+bbox[3]]
            bbox = [str(b) for b in bbox]
            bbox.append("0") # 0 for ignore
            imgid_bboxes_map[id].add(",".join(bbox))

        imgid_path_map[id] = os.path.basename(imgpath)

    return imgid_path_map,imgid_bboxes_map

def gen_pred_img_maps(src_pred_file):
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

    return sorted_ib_bboxes_map
    # return ib_bboxes_map


def get_box_from_str_format(str_box):
    '''
    format then get boxes.
    :param str_box:
    :return:
    '''
    b = str_box.split(',')
    b = [float(bi) for bi in b]
    return b[0:5]

def visionalize_cases(src_annotation_file, src_pred_file,src_img_dir,src_pred_conf, dst_img_save):
    '''
    visionalization for cases.
    :param src_annotation_file:
    :param src_pred_file:
    :param dst_img_save:
    :return:
    '''
    imgid_path_map,imgid_gt_bboxes_map = gen_gt_img_bboxes_map(src_annotation_file)
    imgid_pred_bboxes_map = gen_pred_img_maps(src_pred_file)
    print("len imgid_path_map:%s,  len:%s"%(len(imgid_path_map), len(imgid_gt_bboxes_map)))
    print("len imgid_pred_bboxes_map:%s"%(len(imgid_pred_bboxes_map)))
    all_imgs = len(imgid_path_map)
    # draw gt boxes, as well as pred bboxes
    for id,imgname in imgid_path_map.items():
        if id % 1000 ==0:
            print("now at %s/%s"%(id, all_imgs))

        img_path = os.path.join(src_img_dir, imgname)
        save_img_name = os.path.splitext(os.path.basename(imgname))[0]+"_result.jpg"
        dst_img_path = os.path.join(dst_img_save,save_img_name)
        if not os.path.exists(img_path):
            raise Exception("error %s not exists!!"%(img_path))

        img = cv2.imread(img_path)

        str_gt_boxes = imgid_gt_bboxes_map[id]
        for str_b in str_gt_boxes:
            gtbox = get_box_from_str_format(str_b)
            color = (255,0,0)
            #print("imgId:%s,  bbox: %s,%s,%s,%s"%(id, int(gtbox[0]), int(gtbox[1]), int(gtbox[2]), int(gtbox[3])))
            # img = draw.drawrect(img, tl_point, br_point, color, thickness=1, style='dotted')
            img = cv2.rectangle(img, (int(gtbox[0]), int(gtbox[1])), (int(gtbox[2]),int(gtbox[3])), color, 2)
        try:
            str_pred_boxes = imgid_pred_bboxes_map[id]
        except:
            print("id:%s, len(imgid_pred_bboxes_map):%s"%(id, len(imgid_pred_bboxes_map)))
            continue
        for str_p_b in str_pred_boxes:
            predbox = get_box_from_str_format(str_p_b)
            score  =predbox[4]
            color = (0,255,0)
            if score < float(src_pred_conf):
                continue
            img = cv2.rectangle(img, (int(predbox[0]), int(predbox[1])), (int(predbox[2]),int(predbox[3])), color, 1)

        cv2.imwrite(dst_img_path,img)





def visionalize_bad_cases(src_annotation_file,
                          src_pred_file,
                          src_img_dir,
                          src_pred_conf,
                          src_considered_iou_threshold,
                          src_pred_conf_topK,
                          dst_img_save):
    '''
    visionalization for cases.
    :param src_annotation_file: the gt's annotations file coco format.
    :param src_pred_file: the prediction results.
    :param src_img_dir: src images dir
    :param src_pred_conf: src predictions' conf threshold value. which will filter results below it... only get upper results.
    :param src_considered_iou_threshold: predictions's IOU with groudtruth. results greater than this will be kept.
    :param src_pred_conf_topK: TOPK prediction will be kept.
    :param dst_img_save:
    :return:
    '''
    imgid_path_map,imgid_gt_bboxes_map = gen_gt_img_bboxes_map(src_annotation_file)
    imgid_pred_bboxes_map = gen_pred_img_maps(src_pred_file)
    print("len imgid_path_map:%s,  len:%s"%(len(imgid_path_map), len(imgid_gt_bboxes_map)))
    print("len imgid_pred_bboxes_map:%s"%(len(imgid_pred_bboxes_map)))

    all_imgs = len(imgid_path_map)
    # draw gt boxes, as well as pred bboxes
    imgid_pred_result_map={}
    for id,imgname in imgid_path_map.items():
        if id % 1000 ==0:
            print("now at %s/%s"%(id, all_imgs))

        img_path = os.path.join(src_img_dir, imgname)
        save_img_name = os.path.splitext(os.path.basename(imgname))[0]+"_result.jpg"
        dst_img_path = os.path.join(dst_img_save,save_img_name)
        if not os.path.exists(img_path):
            raise Exception("error %s not exists!!"%(img_path))


        gt_bboxes = [get_box_from_str_format(strb) for strb in imgid_gt_bboxes_map[id]]
        pred_bboxes = [get_box_from_str_format(strb) for strb in imgid_pred_bboxes_map[id]]
        print("id:%s, len gt_bboxes: %s, len pred_bboxes: %s"%(id, len(gt_bboxes), len(pred_bboxes)))
        match_matrix = matrix.calc_match_matrix(src_gtbboxes=gt_bboxes, src_predbboxes=pred_bboxes,src_filter_conf=src_considered_iou_threshold)
        filter_match_matrix = matrix.filter_on_matrix(src_matrix=match_matrix, src_topk=src_pred_conf_topK, src_thresh=src_considered_iou_threshold)
        print("match_matrix: ", match_matrix)
        print("filter_match_matrix: ", filter_match_matrix)
        raise Exception("stop!!")


