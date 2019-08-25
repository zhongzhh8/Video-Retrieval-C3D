import os
import cv2
import numpy as np


def video2frames(video, newdir):
    cap = cv2.VideoCapture(video)
    count = 0
    cnt = 0
    # 29.97/30FPS
    while (cap.isOpened()):
        ret, frame = cap.read()
        if cnt % 1 == 0:  # 30 FPS
            if ret == True:
                # cv2.imshow("frame",frame)
                cv2.imwrite(newdir + '/' + str(count).zfill(4) + ".jpg", frame)
                # cv2.waitKey(0)
                count += 1
            else:
                break
        cnt += 1
    return count


# th14 dataset
# train_fpath = '/home/disk1/wangshaoying/my_video_retrieval/th14/data/val.txt'
# test_fpath = '/home/disk1/wangshaoying/my_video_retrieval/th14/data/test.txt'
# db_fpath = '/home/disk1/wangshaoying/my_video_retrieval/th14/data/db.txt'

# ucf101 20
# train_fpath = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/train1_20.txt'
# test_fpath = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/test1_20.txt'
# db_fpath = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/db1_20.txt'

# ucf101 101
train_fpath = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/train1_101.txt'
test_fpath = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/test1_101.txt'
db_fpath = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/db1_101.txt'

# # JHMDB
# train_fpath = '/home/disk1/wangshaoying/my_video_retrieval/JHMDB/txt/train_10_210.txt'
# test_fpath = '/home/disk1/wangshaoying/my_video_retrieval/JHMDB/txt/test_10_210.txt'
# db_fpath = '/home/disk1/wangshaoying/my_video_retrieval/JHMDB/txt/db_20_420.txt'


fpath = [train_fpath, test_fpath]
for i in range(len(fpath)):
    video_num = 0
    print(fpath[i])
    f = open(fpath[i])
    l = f.readlines()
    f.close()
    # th14 dataset
    # root_dir = '/home/disk1/wangshaoying/my_video_retrieval/th14_5FPS/'  # th14
    # root_dir = '/home/disk1/wangshaoying/my_video_retrieval/JHMDB/frames_30FPS/'  # JHMDB
    root_dir = '/home/disk1/wangshaoying/data/UCF101/'  # UCF101

    for item in l:
        # video_dir = '/home/disk1/wangshaoying/my_video_retrieval/th14/'  # th14
        # video_dir = '/home/disk1/wangshaoying/my_video_retrieval/JHMDB/videos/' # JHMDB
        video_dir = '/home/disk1/wangshaoying/my_video_retrieval/ucf101/UCF101/' # ucf101

        second_dir = root_dir + item.strip().split('/')[0] + '/' #+ item.strip().split('/')[1] + '/' #UCF101
        # second_dir = root_dir + item.strip().split('/')[1] + '/' # JHMDB
        print('second:', second_dir)
        if not os.path.exists(second_dir):
            os.mkdir(second_dir)

        video = video_dir + item.strip().split()[0]
        # newdir = os.path.join(root_dir, item.strip().split()[0].split('.')[0])
        # newdir = os.path.join(second_dir, item.strip().split('/')[2].split()[0].split('.')[0]) # JHMDB
        newdir = os.path.join(second_dir, item.strip().split('/')[1].split()[0].split('.')[0])
        print('newdir:', newdir)
        # if os.path.exists(newdir) == True:
        #     os.system("rm -rf " + newdir)
        os.mkdir(newdir)
        frames_num = video2frames(video, newdir)
        video_num += 1
        print(f'the {video_num}th video: {frames_num}frames')

