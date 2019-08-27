from __future__ import print_function, division
import os


if __name__=='__main__':
    split_dir_path='/home/disk3/a_zhongzhanhui/data/HMDB-51/testTrainMulti_7030_splits'
    labels_dir_path='/home/disk3/a_zhongzhanhui/data/HMDB-51/testTrainMulti_7030_splits/labels'
    classid=0
    train_file=open(os.path.join(labels_dir_path,'train1.txt'),'w')
    test_file = open(os.path.join(labels_dir_path, 'test1.txt'), 'w')
    classID_file = open(os.path.join(labels_dir_path, 'classID.txt'), 'w')

    txt_list=os.listdir(split_dir_path)
    txt_list.sort(key=lambda x: str(x[:-4]))
    for txt_name in txt_list:   #txt_name=brush_hair_test_split1.txt
        if 'split1' not in txt_name:
            continue

        test_str_index=txt_name.index('_test')
        label_name=txt_name[0:test_str_index]

        classID_file.write(label_name+' '+str(classid)+'\n')

        txt_file = open( os.path.join(split_dir_path,txt_name))
        trainsample_cnt= 0
        testsample_cnt =0
        for line in txt_file:
            # video_name = line.strip().split()[0].split('.')[0]  # Depending on your fpath_label file
            video_name = line.strip().split()[0]
            split_id = line.strip().split()[1]  # default for single label, while [1:] for single label
            split_id = int(split_id)
            if split_id==1:
                trainsample_cnt+=1
                train_file.write(label_name+'/'+video_name+' '+str(classid)+'\n')
            elif split_id==2:
                testsample_cnt+=1
                test_file.write(label_name+'/'+video_name+' '+str(classid)+'\n')

        # print(str(trainsample_cnt)+' vs '+str(testsample_cnt))
        txt_file.close()

        classid+=1

    classID_file.close()