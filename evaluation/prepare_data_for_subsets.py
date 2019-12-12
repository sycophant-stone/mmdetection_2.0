import os
import utils.shell as shell

def prepare(src_img_list_file, src_ref_imgs_dir, src_ref_anno_file, dst_img_dir, dst_anno_dir):
    '''prepare for subsets.. from imgid list and reffered datasets..
    '''
    # now copy the specific imgid.. from reffer img dir to dst img dir ..
    with open(src_img_list_file, 'r') as f:
        for line in f.readlines():
            imgname = line.strip() + '.jpg'
            src_img_path = os.path.join(src_ref_imgs_dir, imgname)
            cmd = "cp %s %s"%(src_img_path, dst_img_dir)
            shell.run_system_command(cmd)

    # filter ref coco's anno file to gen dst(target) coco anno file..
