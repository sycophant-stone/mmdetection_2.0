import os
import cv2




if __name__=='__main__':
    # boxes=[
    #     [195,167,61,61],
    #     [0,181,61,73],
    #     [121,121,57,57]
    # ]
    boxes=[
        [ 0,181,61,254],
        [ 121,121,178,178],
        [195,167,256,228]
    ]
    imgpath = "/ssd/hnren/Data/coco_300px_large_labelbox_head/FID_DID_HEAD_CLEAN_0_patches_int/JPEGImages/ch00011_20190329_ch00011_20190329202000.mp4.cut.mp4_000000_crop_1.jpg"
    if not os.path.exists(imgpath):
        raise Exception("%s not exists!"%(imgpath))
    img = cv2.imread(imgpath)
    for b in boxes:
        img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (0,0,255), 1)

    outname = "Gt_alpha_"+os.path.basename(imgpath)
    cv2.imwrite(outname, img)
