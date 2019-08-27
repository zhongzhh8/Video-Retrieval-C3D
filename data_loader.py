import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image


class CustomDataset(data.Dataset):

    def __init__(self, root_folder, fpath_label, transform=None,num_frames=32):  #  fpath_label.txt: frames_dir video_label

        f = open(fpath_label)
        l = f.readlines()
        f.close()
        # print l
        fpaths = list()
        labels = list()
        for item in l:
            path = item.strip().split()[0].split('.')[0]  # Depending on your fpath_label file
            label = item.strip().split()[1]  # default for single label, while [1:] for single label
            label = int(label)
            fpaths.append(path)
            labels.append(label)

        self.root_folder = root_folder
        self.fpaths = fpaths
        self.labels = labels
        self.label_size = len(self.labels)
        self.transform = transform
        self.num_frames=num_frames


    def __getitem__(self, index):

        label = self.labels[index]
        ########## can use cv2 to process frames...#########
        frames_dir = self.root_folder + self.fpaths[index]
        l_ = os.listdir(frames_dir)
        l_.sort(key=lambda x: str(x[:-4]))

        frames_length = self.num_frames

        l = [l_[int(round(i * len(l_) / float(frames_length)))] for i in range(frames_length)]

        assert len(l) == self.num_frames
        frames_array = np.zeros((frames_length, 3, 112, 112), dtype=np.float32)

        for i in range(frames_length):
            # frame=cv2.imread(frames_dir+"/"+l[i])
            # frame=cv2.resize(frame,(171,128))k
            frame = Image.open(frames_dir + "/" + l[i]).convert("RGB")
            # cv2.imshow("training frames",frame)
            # cv2.waitKey(1)
            if not self.transform == None:
                frame = self.transform(frame)
                frame = frame.numpy()
            frames_array[i, :, :, :] = frame
            # print frames_array[i,:,:,:].sum()
        frames_array = frames_array.transpose((1, 0, 2, 3))
        # print frames_array
        ##########################################################

        label = torch.tensor(label)
        frames = torch.tensor(frames_array)
        return frames, label


    def __len__(self):
        return len(self.fpaths)
