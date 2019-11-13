import os
import cv2
import argparse
import numpy as np
from pycocotools.coco import COCO

def get_cat(coco):
    '''
    get categories
    :param coco:
    :return: list for categories' names.
    '''
    cats = coco.loadCats(coco.getCatIds())
    names = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(names)))
    return names

def get_img_ids(coco, src_cat_names):
    '''
    get imgids for cat names
    :param src_cat_name:  a list for cat names
    :return:
    '''
    catIds = coco.getCatIds(catNms=src_cat_names)
    imgIds = coco.getImgIds(catIds=catIds)

def get_img_by_id(coco, src_img_id):
    '''
    get img for img_id
    :param src_img_id: list
    :return:
    '''
    img = coco.loadImgs(src_img_id)
    return img
def get_annoids_by_imgid(coco,src_cat_id, src_img_id):
    '''
    get anno-ids by img id
    :param coco:
    :param src_img_id:
    :return:
    '''

    anno_id_set = coco.getAnnIds(imgIds=src_img_id, catIds=[1], iscrowd=None)
    return anno_id_set

def get_anno_by_id(coco, anno_id):
    '''
    get anno by id
    :param coco:
    :param anno_id:
    :return:
    '''
    anno0 = coco.loadAnns(anno_id)
    return anno0

def bbox_convert_to_squarebox(src_bbox):
    '''
    convert non-squarebox to squarebox
    :param src_bbox:
    :return:
    '''
    xmin,ymin,xmax,ymax = src_bbox
    w_raw = xmax - xmin + 1
    h_raw = ymax - ymin + 1
    sz = int(max(w_raw, h_raw) * 0.62)
    x = int(xmin + (w_raw - sz) * 0.5)
    y = int(ymin + h_raw - sz)
    new_xmin = x
    new_ymin = y
    new_xmax = x + sz - 1
    new_ymax = y + sz - 1
    # print("x,y,sz:(%d,%d,%d)",x,y,sz)
    new_xmin = new_xmin if new_xmin >= 0 else 0
    new_ymin = new_ymin if new_ymin >= 0 else 0
    new_xmax = new_xmax if new_xmax < w else w - 1
    new_ymax = new_ymax if new_ymax < h else h - 1

def xytoxcyc(x,y,sz):
    '''
    convert x,y to xc, yc
    :param x:
    :param y:
    :param sz:
    :return:
    '''
    xc = x+(sz/2)
    yc = y+(sz/2)
    return xc,yc

def xcyctoxy(xc,yc,expand_sz,width,height):
    '''
    xc,yc to x,y
    :param xc:
    :param yc:
    :param expand_sz:
    :param width:
    :param height:
    :return:
    '''
    xmin = xc - expand_sz/2
    ymin = yc - expand_sz/2
    xmax = xc + expand_sz/2
    ymax = yc + expand_sz/2

    xmin = np.clip(xmin, 0, width-1)
    xmax = np.clip(xmax, 0, width-1)
    ymin = np.clip(ymin, 0, height-1)
    ymax = np.clip(ymax, 0, height-1)

    return xmin,ymin,xmax,ymax

def GET_BARE_STRING(filepath):
    '''
    get file's bare name string.
    :param filepath: xxx/xxxx/zzz.jpg
    :return: zzz
    '''
    if not os.path.exists(filepath):
        raise Exception("%s not exists"%(filepath))
    return os.path.splitext(os.path.basename(filepath))[0]

def data_preprocess(src_coco_anno_file,
                    out_images_dir,
                    out_anno_dir):
    '''
    coco dataset's preprocessing. including clip, crop.
    :param src_coco_anno_file:
    :return:
    '''
    # initialize COCO api for instance annotations
    coco=COCO(src_coco_anno_file)

    names = get_cat(coco)
    imgids = get_img_ids(coco,names)
    img_0 = get_img_by_id(coco, 0)


    filepath = img_0[0]['file_name']
    width = img_0[0]['width']
    height = img_0[0]['height']
    id = img_0[0]['id']
    img_barename = GET_BARE_STRING(filepath)
    img = cv2.imread(filepath)
    src_cat_id = [1] # head's coco datasets, there is only one class.
    anno_id = get_annoids_by_imgid(coco,src_cat_id, id)

    imgid_annobox_map = {}
    imgid_expandbox_map = {}
    for annoid in anno_id:
        anno = get_anno_by_id(coco, annoid)
        #print("anno:", anno)
        box = anno['bbox']
        xmin,ymin,w,h = box
        xmax = xmin + w
        ymax = ymin + h

        xc,yc = xytoxcyc(xmin,ymin,min(w,h))
        # expand to 300,300
        expand_sz = 300
        exmin, eymin, exmax,eymax = xcyctoxy(xc,yc,expand_sz,width,height)
        imgid = 0
        annobox = [xmin,ymin,xmax,ymax]
        expandbox=[exmin,eymin,exmax,eymax]

        if imgid not in imgid_annobox_map:
            imgid_annobox_map[imgid] = set([])
        imgid_annobox_map[imgid].add(annobox)

        if imgid not in imgid_expandbox_map:
            imgid_expandbox_map[imgid] = set([])
        imgid_expandbox_map[imgid].add(expandbox)

    for imgid, eboxes in imgid_expandbox_map.items():
        for ebox in eboxes:
            exmin,eymin,exmax,eymax = ebox
            img_croped = img[exmin:exmax, eymin:eymax]
            img_croped_name = os.path.join(out_images_dir, "%s_crop.jpg"%(img_barename))
            cv2.imwrite(img_croped, img_croped_name)
            # gen xmlbox for this crop img
            gen_xmlbox(src_crop_region=, src_img_id=, src_cat_id= ,)



    #ioregion_croped = img[img_ymn:img_ymx, img_xmn:img_xmx]


def gen_fonders(input_outdir):
    '''
    gen fonders for output files.
    :param input_outdir:
    :return:
    '''
    out_images_dir = os.path.join(input_outdir, "images")
    out_anno_dir = os.path.join(input_outdir,"annotations")
    if not os.path.exists(out_images_dir):
        os.mkdir(out_images_dir)
    if not os.path.exists(out_anno_dir):
        os.mkdir(out_anno_dir)

    return out_images_dir, out_anno_dir

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--anno', type=str, help='instance_*.json filepath', required=True)
    parse.add_argument('--outdir', type=str, help='resized file out dir ,including cropped images and resized anno json file', required=True)
    _args = parse.parse_args()
    input_anno = _args.anno
    input_outdir = _args.outdir
    out_images_dir, out_anno_dir =gen_fonders(input_outdir)
    data_preprocess(src_coco_anno_file=input_anno,out_images_dir, out_anno_dir)