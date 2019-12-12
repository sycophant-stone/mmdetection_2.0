mmdetections
=============
	TODO#
	bad case 分析
		建立一个通过imgid来做eval的yaml..
			依赖
				python pipe.py --yaml pipe_config/vision_cases.yaml --data_dir /ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out_beta --gtfile /ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json --imgs /ssd/hnren/Data/coco_300px_clean_head_beta/JPEGImages --predfile /ssd/hnren/2nd/zccstig_fsaf/mmdetection/2.pkl.json --pred_conf 0.45
				
			创建一个新的..
				pipe_config/show_info_by_imgid.yaml
				python pipe.py --yaml pipe_config/show_info_by_imgid.yaml --data_dir /ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out_beta --gtfile 	/ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json --imgs 	/ssd/hnren/Data/coco_300px_clean_head_beta/JPEGImages --predfile /ssd/hnren/2nd/zccstig_fsaf/mmdetection/2.pkl.json --pred_conf 0.45 --imgid ch00005_20190214_ch00005_20190214115838.mp4.cut.mp4_010500_crop6
		

		列出所有的bad 图片
				
			python pipe.py --yaml pipe_config/vision_cases.yaml \
			--data_dir /ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out_beta \
			--gtfile /ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json \
			--imgs /ssd/hnren/Data/coco_300px_clean_head_beta/JPEGImages \
			--predfile /ssd/hnren/2nd/zccstig_fsaf/mmdetection/2.pkl.json \
			--pred_conf 0.45
		
		复现某一张id的bad情况..
			python pipe.py --yaml pipe_config/show_info_by_imgid.yaml --data_dir /ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out_beta --gtfile /ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json --imgs /ssd/hnren/Data/coco_300px_clean_head_beta/JPEGImages --predfile /ssd/hnren/2nd/zccstig_fsaf/mmdetection/2.pkl.json --pred_conf 0.45 --imgid ch00005_20190214_ch00005_20190214115838.mp4.cut.mp4_010500_crop6
		
		列出所有的:
			pipe_config/show_info_by_imgid_list.yaml
			
				python pipe.py --yaml pipe_config/show_info_by_imgid_list.yaml --data_dir /ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out_beta --gtfile /ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json --imgs /ssd/hnren/Data/coco_300px_clean_head_beta/JPEGImages --predfile /ssd/hnren/2nd/zccstig_fsaf/mmdetection/2.pkl.json --pred_conf 0.45 --bad_imgs_dir /ssd/hnren/2nd/zccstig_fsaf/mmdetection/eva_out_beta/dataset/eva/result__0.20/bad_case_imgs
	
	
	复现Resnet50.. Resnet18..
	通过distill来把Resnet50 提炼到Resnet18
	
	
	