import os
import cv2
import argparse
import evaluation.matrix as matrix
import evaluation.coco_gt_reader as coco_gt_reader
import evaluation.pred_reader as pred_reader
from evaluation.data_reader import get_box_from_str_format, listdata_saveto_csv_data, csv_file_to_listdata


def gen_cases_for_visionalization(src_annotation_file, src_pred_file,src_img_dir,src_pred_conf, dst_img_save):
    '''
    visionalization for cases.
    :param src_annotation_file:
    :param src_pred_file:
    :param dst_img_save:
    :return:
    '''
    cgr = coco_gt_reader.GtReaderCoCo(src_annotation_file)
    imgid_path_map,imgid_gt_bboxes_map = cgr.gen_gt_img_bboxes_map()

    pr = pred_reader.PredReader(src_pred_file)
    imgid_pred_bboxes_map = pr.gen_pred_img_maps()

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





def statistic_bad_cases(src_annotation_file,
                          src_pred_file,
                          src_img_dir,
                          src_pred_conf,
                          src_iou_high,
                          src_iou_low,
                          src_pred_conf_topK,
                          dst_bad_cases_record_csv_file,
                        ):
    '''
    visionalization for cases.
    :param src_annotation_file: the gt's annotations file coco format.
    :param src_pred_file: the prediction results.
    :param src_img_dir: src images dir
    :param src_pred_conf: src predictions' conf threshold value. which will filter results below it... only get upper results.
    :param src_iou_high: predictions's IOU with groudtruth. results greater than this will be kept.
    :param src_pred_conf_topK: TOPK prediction will be kept.
    :param dst_img_save:
    :return:
    '''

    cgr = coco_gt_reader.GtReaderCoCo(src_annotation_file)
    imgid_path_map,imgid_gt_bboxes_map = cgr.gen_gt_img_bboxes_map()

    pr = pred_reader.PredReader(src_pred_file)
    imgid_pred_bboxes_map = pr.gen_pred_img_maps()

    print("len imgid_path_map:%s,  len:%s"%(len(imgid_path_map), len(imgid_gt_bboxes_map)))
    print("len imgid_pred_bboxes_map:%s"%(len(imgid_pred_bboxes_map)))

    all_imgs = len(imgid_path_map)
    # draw gt boxes, as well as pred bboxes
    imgid_pred_result_map={}
    for id,imgname in imgid_path_map.items():
        if id % 1000 ==0:
            print("now at %s/%s"%(id, all_imgs))

        gt_bboxes = [get_box_from_str_format(strb) for strb in imgid_gt_bboxes_map[id]]
        pred_bboxes = [get_box_from_str_format(strb) for strb in imgid_pred_bboxes_map[id]]
        print("id:%s, len gt_bboxes: %s, len pred_bboxes: %s"%(id, len(gt_bboxes), len(pred_bboxes)))
        match_matrix = matrix.calc_match_matrix(src_gtbboxes=gt_bboxes, src_predbboxes=pred_bboxes,src_filter_conf=src_iou_high)
        filter_match_matrix = matrix.filter_on_matrix(src_matrix=match_matrix,
                                                      src_topk=src_pred_conf_topK,
                                                      src_thresh_high=src_iou_high,
                                                      src_thresh_low=src_iou_low,
                                                      src_predbboxes=pred_bboxes,
                                                      src_conf_thresh_high=0.8,
                                                      src_conf_thresh_low=0.01)
        statis_matrix_results = matrix.statistic_on_matrix(src_matrix=filter_match_matrix)
        #print("match_matrix: ", match_matrix)
        #print("filter_match_matrix: ", filter_match_matrix)
        #print("statis_matrix_results ", statis_matrix_results )
        # raise Exception("stop!!")
        imgid_pred_result_map[id] = statis_matrix_results

    # print("imgid_pred_result_map ", imgid_pred_result_map )
    # those are imgid { annoid(objid):"tp,fp,fn", ...}

    # for imgid, matched_matrix in imgid_pred_result_map.items():
    #    print("imgid:%s, matched_matrix:%s"%(imgid, match_matrix))
    #    raise Exception("stop!!")

    # print(imgid_pred_result_map[0])
    # print("type of imgid_pred_result_map: ", type(imgid_pred_result_map))

    fpfn_imgid_list = []
    for id, mres in imgid_pred_result_map.items():
        for gt_obj_ind,tfpn_str in mres.items():
            tp,fp,fn = matrix.get_tfpn_from_string(tfpn_str)
            if tp ==0:
                fpfn_imgid_list.append(id)
                break

    print("fpfn_imgid_list: ", fpfn_imgid_list)
    print("len fpfn_imgid_list: ", len(fpfn_imgid_list))

    listdata_saveto_csv_data(fpfn_imgid_list, dst_bad_cases_record_csv_file)



def Test_get_pred_gt_info_by_imgid(src_imgid=16185):
    imgid = int(src_imgid)
    src_annotation_file = "/ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out//dataset/eva/source/instances_train.json"
    src_pred_file = "/ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out//dataset/eva/source/eva_r18.pkl.json"
    cgr = coco_gt_reader.GtReaderCoCo(src_annotation_file)
    imgid_path_map,imgid_gt_bboxes_map = cgr.gen_gt_img_bboxes_map()

    pr = pred_reader.PredReader(src_pred_file)
    imgid_pred_bboxes_map = pr.gen_pred_img_maps()

    pred_boxes_for_this_id = pr.get_bboxes_by_imgid(imgid=imgid)
    print("id:%s, pred_boxes_for_this_id: %s"%(imgid, pred_boxes_for_this_id))
    objids = cgr.get_annoidlist_for_specific_imgid(imgid)
    print("objids: ", objids)
    for objinx in objids:
        print("objinx:",objinx)
        gt_boxes_for_this_id = cgr.get_bbox_by_imgid_annoid(imgid=imgid, annoid=objinx)
        print("objid:%s, gt_boxes_for_this_id: %s"%(objinx, gt_boxes_for_this_id))

    gt_bboxes = [get_box_from_str_format(strb) for strb in imgid_gt_bboxes_map[imgid]]
    pred_bboxes = [get_box_from_str_format(strb) for strb in imgid_pred_bboxes_map[imgid]]
    src_iou_high = 0.5
    src_iou_low = 0.5
    src_pred_conf_topK = 10
    match_matrix = matrix.calc_match_matrix(src_gtbboxes=gt_bboxes, src_predbboxes=pred_bboxes,
                                            src_filter_conf=src_iou_high)
    print("imgid:%s, match_matrix:%s"%(imgid, match_matrix))
    filter_match_matrix = matrix.filter_on_matrix(src_matrix=match_matrix,
                                                  src_topk=src_pred_conf_topK,
                                                  src_thresh_high=src_iou_high,
                                                  src_thresh_low=src_iou_low,
                                                  src_predbboxes=pred_bboxes,
                                                  src_conf_thresh_high=0.8,
                                                  src_conf_thresh_low=0.01)

    print("imgid:%s, filter_match_matrix:%s"%(imgid, filter_match_matrix))
    statis_matrix_results = matrix.statistic_on_matrix(src_matrix=filter_match_matrix)
    print("imgid:%s, statis_matrix_results:%s"%(imgid, statis_matrix_results))


def visionalize_bad_case(src_img_dir,
                         src_annotation_file,
                         src_pred_file,
                         src_bad_case_csv_file,
                         src_visual_pred_conf,
                         dst_bad_case_imgs_dir,
                         dst_bad_case_info_file):
    '''
    visionalize bad cases.

    :param src_bad_case_csv_file:  1,2,3,...
    first convert to [1,2,3...] , list.
    :param dst_bad_case_imgs_dir:
    :return:
    '''
    bc_imgid_list = csv_file_to_listdata(src_bad_case_csv_file)
    print("bc_imgid_list: ", bc_imgid_list)
    cgr = coco_gt_reader.GtReaderCoCo(src_annotation_file)
    imgid_path_map,imgid_gt_bboxes_map = cgr.gen_gt_img_bboxes_map()

    pr = pred_reader.PredReader(src_pred_file)
    imgid_pred_bboxes_map = pr.gen_pred_img_maps()

    print("len imgid_path_map:%s,  len:%s"%(len(imgid_path_map), len(imgid_gt_bboxes_map)))
    print("len imgid_pred_bboxes_map:%s"%(len(imgid_pred_bboxes_map)))
    bc_info_p = open(dst_bad_case_info_file, 'w')
    for bc_imgid in bc_imgid_list:
        # print("case#%s"%(bc_imgid))
        gt_bboxes = [get_box_from_str_format(strb) for strb in imgid_gt_bboxes_map[bc_imgid]]
        pred_bboxes = [get_box_from_str_format(strb) for strb in imgid_pred_bboxes_map[bc_imgid]]
        imgname = imgid_path_map[bc_imgid]
        img_path = os.path.join(src_img_dir, imgname)
        save_img_name = os.path.splitext(os.path.basename(imgname))[0] + "_result.jpg"
        dst_img_path = os.path.join(dst_bad_case_imgs_dir, save_img_name)
        if not os.path.exists(img_path):
            raise Exception("error %s not exists!!" % (img_path))

        img = cv2.imread(img_path)

        str_gt_boxes = imgid_gt_bboxes_map[bc_imgid]
        for str_b in str_gt_boxes:
            gtbox = get_box_from_str_format(str_b)
            color = (255, 0, 0)
            # print("imgId:%s,  bbox: %s,%s,%s,%s"%(id, int(gtbox[0]), int(gtbox[1]), int(gtbox[2]), int(gtbox[3])))
            # img = draw.drawrect(img, tl_point, br_point, color, thickness=1, style='dotted')
            img = cv2.rectangle(img, (int(gtbox[0]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[3])), color, 2)
        try:
            str_pred_boxes = imgid_pred_bboxes_map[bc_imgid]
        except:
            print("id:%s, len(imgid_pred_bboxes_map):%s" % (bc_imgid, len(imgid_pred_bboxes_map)))
            continue
        satisfied_conf_list = []
        for str_p_b in str_pred_boxes:
            predbox = get_box_from_str_format(str_p_b)
            score = predbox[4]
            color = (0, 255, 0)
            if score > float(src_visual_pred_conf):
                img = cv2.rectangle(img, (int(predbox[0]), int(predbox[1])), (int(predbox[2]), int(predbox[3])), color, 1)
                satisfied_conf_list.append(str_p_b.replace(',','-'))

        bc_info_p.write("%s,%s,%s\n"%(bc_imgid, imgid_path_map[bc_imgid], '__'.join(satisfied_conf_list)))

        cv2.imwrite(dst_img_path, img)

    bc_info_p.close()

if __name__=='__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--imgid', type=str, help='images id', required=True)
    _args = parse.parse_args()
    imgid = _args.imgid;

    Test_get_pred_gt_info_by_imgid(src_imgid=imgid)