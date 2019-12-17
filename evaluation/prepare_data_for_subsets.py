import os
import utils.shell as shell
import utils.coco_anno_reader as cocoreder


def prepare(src_img_list_file, src_ref_imgs_dir, src_ref_anno_file, dst_img_dir, dst_anno_dir):
    '''prepare for subsets.. from imgid list and reffered datasets..
    '''
    # now copy the specific imgid.. from reffer img dir to dst img dir ..
    with open(src_img_list_file, 'r') as f:
        for line in f.readlines():
            imgname = line.strip() + '.jpg'
            src_img_path = os.path.join(src_ref_imgs_dir, imgname)
            cmd = "cp %s %s" % (src_img_path, dst_img_dir)
            shell.run_system_command(cmd)

    # convert str img id to number ..
    cocoanno = cocoreder.CocoAnno(src_coco_anno_file=src_ref_anno_file)
    sub_img_id_list = []
    imgname_imgid_map = {}
    img_id_list = cocoanno.get_img_id_list()
    for id, imgid in enumerate(img_id_list):
        img_info = cocoanno.get_img_info_by_img_ids(imgid)
        img_name = img_info[0]['file_name']
        img_name = os.path.splitext(os.path.basename(img_name))[0]
        imgname_imgid_map[img_name] = id
        sub_img_id_list.append(id)

    # filter ref coco's anno file to gen dst(target) coco anno file..
    main_set_barename = os.path.splitext(os.path.basename(src_ref_anno_file))[0]
    subset_name = main_set_barename + '_subset.json'
    dst_anno_file_name = os.path.join(dst_anno_dir, subset_name)
    cocoreder.gen_sub_datasets(
        src_imgid_list=sub_img_id_list,
        src_ref_coco_file=src_ref_anno_file,
        dst_subset_coco_file=dst_anno_file_name
    )
