import os
import cv2
import mmcv
import argparse
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result
import mmdet.ops.nms as mm_nms


#  test a list of images and write the results to image files
# mgs = ['test1.jpg', 'test2.jpg']
# or i, result in enumerate(inference_detector(model, imgs):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

def GET_FILE_BARENAME(filename):
    if not os.path.exists(filename):
        raise Exception("%s not exists"%(filename))
    return os.path.splitext(os.path.basename(filename))[0]


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

'''
this script will test one picture, and draw those detection's predictions bounding boxes on the picture.
'''
if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--src_config_file', type=str, help='model config file path', required=True)
    parse.add_argument('--src_checkpoint_file', type=str, help='model checkpoint file path', required=True)
    parse.add_argument('--src_picture_file', type=str, help='target pictures path', required=True)
    parse.add_argument('--show_info', type=bool, help='whether show more info', required=False,default=False)

    _args = parse.parse_args()
    input_src_config_file       = _args.src_config_file
    input_src_checkpoint_file   = _args.src_checkpoint_file
    input_src_picture_file      = _args.src_picture_file
    input_show_info             = _args.show_info

    test_and_save(src_config_file = input_src_config_file,
                  src_checkpoint_file=input_src_checkpoint_file,
                  src_picture_file=input_src_picture_file,
                  input_show_info=input_show_info)
