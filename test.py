import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data_loader import CustomDataset
from backbone_3dresnet import resnet18
from temporal_attention import TemporalAttention, TemporalAvgPool, TemporalMaxPool
from hash_layer import HashLayer
from utils import *


def load_data(root_folder, fpath_label, batch_size, shuffle=False, num_workers=4, long=False):
    transform = transforms.Compose([
        # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
        transforms.Resize((128, 171)),
        transforms.CenterCrop((112, 112)),  # Center
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])

    data_ = CustomDataset(root_folder=root_folder,
                          fpath_label=fpath_label,
                          transform=transform,
                          long=long)

    # torch.utils.data.DataLoader
    loader_ = data.DataLoader(
        dataset=data_,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=shuffle,  # shuffle
        num_workers=num_workers)  # multi thread

    return loader_


def visualization(frames):
    imgs = frames.numpy().transpose((0, 2, 3, 4, 1))
    img0 = imgs[0][0]
    img2 = imgs[2][0]
    cv2.imshow("img0", img0)
    cv2.waitKey(1)
    cv2.imshow("img2", img2)
    cv2.waitKey(1)


def load_state(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location="cpu")
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith("module.")):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


if __name__ == "__main__":
    ### set parameter
    gpu = "cuda:0"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")  # device configuration
    label_size = 1
    batch_size = 12  # 128
    hash_length = 48  # 48
    num_classes = 20  # 101
    num_epochs = 140
    k = 100
    attention = False

    ### set network 
    backbone = resnet18()
    temporal_attention = TemporalAttention(temporal=30, reduction=1, multiply=False)
    temporal_avg_pool = TemporalAvgPool().to(device)
    temporal_max_pool = TemporalMaxPool().to(device)
    hash_layer = HashLayer(hash_length, num_classes)
    backbone.to(device)
    temporal_attention.to(device)
    hash_layer.to(device)
    
    backbone.eval()
    temporal_attention.eval()
    hash_layer.eval()

    ### set input/output
    long_root_folder = "/home/disk1/wangshaoying/my_video_retrieval/th14/"
    long_db_fpath_label = "/home/disk1/wangshaoying/my_video_retrieval/th14/data/db.txt"
    # long_db_fpath_label = "/home/disk1/wangshaoying/my_video_retrieval/th14/data/test.txt"
    root_folder = "/home/disk1/wangshaoying/my_video_retrieval/ucf101/ucf101_30FPS/"
    test_fpath_label ="/home/disk1/wangshaoying/my_video_retrieval/ucf101/test1_20.txt"
    train_fpath_label = "/home/disk1/wangshaoying/my_video_retrieval/ucf101/train1_20.txt"

    test_loader = load_data(root_folder, test_fpath_label, batch_size*10, False, 16, False)
    db_loader = load_data(long_root_folder, long_db_fpath_label, batch_size, False, 16, True)
    train_loader = load_data(root_folder, train_fpath_label, batch_size*10, False, 16, False)

    max_mAP = 0.
    max_accuracy = 0.
    max_train_mAP = 0.
    max_topk_mAP = 0.
    for epoch in range(int(num_epochs / 10)):
        backbone_path_dir = '/home/disk1/wangshaoying/video_src/video_gl/log/without_attention/backbone/' + f'{(epoch+1)*10}epoch.pth'
        hash_path_dir = '/home/disk1/wangshaoying/video_src/video_gl/log/without_attention/hash_layer/' + f'{(epoch+1)*10}epoch.pth'
        attention_path_dir = '/home/disk1/wangshaoying/video_src/video_gl/log/without_attention/temporal_attention/' + f'{(epoch+1)*10}epoch.pth'

        backbone = torch.load(backbone_path_dir, map_location=gpu)
        hash_layer = torch.load(hash_path_dir, map_location=gpu)
        temporal_attention = torch.load(attention_path_dir, map_location=gpu)

        train_binary, train_labels = inference(train_loader, backbone, temporal_avg_pool, hash_layer, hash_length, device)
        test_binary, test_labels = inference(test_loader, backbone, temporal_avg_pool, hash_layer, hash_length, device)
        if attention:
            db_binary, db_labels = attention_inference(db_loader, backbone, temporal_attention, temporal_avg_pool, hash_layer, hash_length, device)
        else:
            db_binary, db_labels = inference(db_loader, backbone, temporal_avg_pool, hash_layer, hash_length, device)

        train_mAP = compute_MAP(db_binary, db_labels, train_binary, train_labels)
        train_mAP0 = compute_MAP(train_binary, train_labels, train_binary, train_labels)
        mAP = compute_MAP(db_binary, db_labels, test_binary, test_labels)
        topk_mAP = compute_topk_mAP(db_binary, db_labels, test_binary, test_labels, k)
        
        print(f'[{(epoch+1)*10}/{num_epochs}] '
              f'train_mAP0:{train_mAP0:0.4f} '
              f'train_mAP:{train_mAP:0.4f} '
              f'mAP:{mAP:0.4f} top{k}_mAP:{topk_mAP:0.4f}')
        
        if max_train_mAP < train_mAP:
            max_train_mAP = train_mAP
        if max_mAP < mAP:
            max_mAP = mAP
        if max_topk_mAP < topk_mAP:
            max_topk_mAP = topk_mAP
    
    print(f'max_train_mAP:{max_train_mAP:0.4f} '
          f'max_mAP:{max_mAP:0.4f} '
          f'max_topk_mAP:{max_topk_mAP:0.4f}')

    # print(len(test_loader))
    # f = open("./hashcodes/test_hashcode.txt", "w+")
    # f_ = open("./hashcodes/test_label.txt", "w+")
    # total = 0
    # correct = 0
    # with torch.no_grad():
    #     for i, (frames, labels) in enumerate(test_loader):
    #         frames = frames.to(device)  # to gpu if possible
    #         labels = labels.to(device)  # to gpu if possible
    #
    #         features = backbone(frames)
    #         h, c = hash_layer(features)
    #         _, predicted = torch.max(c.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum()
    #
    #         hashcode_array = h.cpu().numpy()
    #         for row in range(hashcode_array.shape[0]):
    #             record = " ".join(map(str, list(hashcode_array[row]))) + '\n'
    #             f.write(record)
    #             f_.write(str(int(labels[row].item())) + " \n")
    #     print('Accuracy of the short query: {} %'.format(100 * correct / total))
    #
    #     f.close()
    #     f_.close()
    #
    # print(len(long_test_loader))
    # f = open("./hashcodes/long_db_hashcode.txt", "w+")
    # f_ = open("./hashcodes/long_db_label.txt", "w+")
    # f__ = open("./attention_weight.txt", "w+")
    # total = 0
    # correct = 0
    # with torch.no_grad():
    #     for i, (long_frames, long_labels) in enumerate(long_test_loader):
    #         long_frames = long_frames.to(device)  # to gpu if possible
    #         long_labels = long_labels.to(device)  # to gpu if possible
    #
    #         long_features = backbone(long_frames)
    #         # print long_features.size()
    #         # long_features_atten = temporal_attention(long_features)
    #
    #         w = temporal_attention(long_features)
    #         long_features_atten = long_features * w
    #         # long_features_atten = long_features * (1 + w)
    #
    #         long_features_avg = temporal_avg_pool(long_features_atten)  # long_features_atten
    #         # long_features_max = temporal_max_pool(long_features)
    #         h, c = hash_layer(long_features_avg)
    #
    #         _, predicted = torch.max(c.data, 1)
    #         total += long_labels.size(0)
    #         correct += (predicted == long_labels).sum()
    #
    #         hashcode_array = h.cpu().numpy()
    #         for row in range(hashcode_array.shape[0]):
    #             record = " ".join(map(str, list(hashcode_array[row]))) + '\n'
    #             f.write(record)
    #             f_.write(str(int(long_labels[row].item())) + " \n")
    #         w_array = torch.squeeze(w).cpu().numpy()
    #         for row in range(w_array.shape[0]):
    #             record = " ".join(map(str, list(w_array[row]))) + '\n'
    #             f__.write(record)
    #     print('Accuracy of the long clip: {} %'.format(100 * correct / total))
    #
    #     f.close()
    #     f_.close()
    #     f__.close()
