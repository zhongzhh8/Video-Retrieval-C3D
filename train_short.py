import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from data_loader import CustomDataset
# from multi_stream_loader import CustomDataset
from backbone_3dresnet import resnet18
from triplet_loss import TripletLoss
from temporal_attention import TemporalAttention, TemporalAvgPool, TemporalMaxPool
from hash_layer import HashLayer
from quantization_loss import QuantizationLoss  # , QuantizationLoss1
from max_entropy_loss import MaxEntropyLoss
import time
import os
import torch.nn.functional as F




def load_data(root_folder, fpath_label, batch_size, shuffle=True, num_workers=16, long=False):
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


# def load_multi_stream(root_folder, fpath_label, long_root_folder, long_fpath_label, batch_size, shuffle=True,
#                       num_workers=4):
#     transform = transforms.Compose([transforms.ToPILImage(),
#                                     # Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
#                                     # transforms.Resize((171,128)),
#                                     transforms.CenterCrop((112, 112)),  # Center
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])
#
#     data_ = CustomDataset(root_folder=root_folder,
#                           fpath_label=fpath_label,
#                           long_root_folder=long_root_folder,
#                           long_fpath_label=long_fpath_label,
#                           transform=transform)
#
#     # torch.utils.data.DataLoader
#     loader_ = data.DataLoader(
#         dataset=data_,  # torch TensorDataset format
#         batch_size=batch_size,  # mini batch size
#         shuffle=shuffle,  # shuffle
#         num_workers=num_workers)  # multi thread
#
#     return loader_


def load_state(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location="cpu")["state_dict"]
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


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == "__main__":
    ### set parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device configuration
    label_size = 1
    batch_size = 3 * 4  # 4*4
    num_epochs = 200  # 100
    step_size = 10  # 40
    learning_rate = 0.001
    hash_length = 64
    margin = 8
    num_classes = 101  # 101

    ### set network and optimizer
    backbone = resnet18()
    load_state(backbone, "/home/disk1/wangshaoying/my_video_retrieval/pretrained/resnet-18-kinetics.pth")
    backbone.to(device)
    backbone = torch.nn.DataParallel(backbone, device_ids=[0, 1])  # for multi gpu
    temporal_avg_pool = TemporalAvgPool().to(device)
    temporal_max_pool = TemporalMaxPool().to(device)
    hash_layer = HashLayer(hash_length, num_classes).to(device)
    triplet_loss = TripletLoss(margin, device).to(device)
    crossentropy = nn.CrossEntropyLoss().to(device)

    params = list(backbone.parameters()) + list(hash_layer.parameters())
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size)

    ### set input/output
    root_folder = "/home/disk1/wangshaoying/data/ucf101/videos/"
    train_fpath_label = "/home/disk1/wangshaoying/data/ucf101/train_test_list/train1_101.txt"

    train_loader = load_data(root_folder, train_fpath_label, int(batch_size * 10), True, num_workers=8, long=False)
    train_loader_iter = iter(cycle(train_loader))

    backbone.train()
    hash_layer.train()

    backbone_path_dir = '/home/disk1/wangshaoying/video_src/video_gl/log/ucf101_101/backbone_32frames_s10/'
    hash_path_dir = '/home/disk1/wangshaoying/video_src/video_gl/log/ucf101_101/hash_layer_32frames_s10/'
    if not os.path.exists(backbone_path_dir):
        os.mkdir(backbone_path_dir)
        os.mkdir(hash_path_dir)

    total_step = len(train_loader) # total_step in one epoch
    start_time = time.time()
    for epoch in range(num_epochs):
        scheduler.step()
        epoch_loss = epoch_loss_h = epoch_loss_c = 0.

        for i in range(total_step):
            ''' 
            try:
                frames, labels = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                frames, labels = next(train_loader_iter)
            try:
                long_frames, long_labels = next(long_train_loader_iter)
            except StopIteration:
                long_train_loader_iter = iter(long_train_loader)
                long_frames, long_labels = next(long_train_loader_iter)
            '''
            frames, labels = next(train_loader_iter)
            frames = frames.to(device)  # to gpu if possible
            labels = labels.to(device)  # to gpu if possible

            features = backbone(frames)
            features_avg = temporal_avg_pool(features)

            h, c = hash_layer(features_avg)
            loss_h = triplet_loss(h, labels)
            loss_c = crossentropy(c, labels)
            loss = loss_h + loss_c
            #
            # if loss == 0:
            #     continue

            ### Backward and optimize
            start = time.clock()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.clock()

            epoch_loss += loss.item()
            epoch_loss_h += loss_h.item()
            epoch_loss_c += loss_c.item()

        avg_loss = epoch_loss / total_step
        avg_loss_h = epoch_loss_h / total_step
        avg_loss_c = epoch_loss_c / total_step
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        print(f'[{epoch}/{num_epochs}] loss:{avg_loss:0.2f}  loss_h:{avg_loss_h:0.2f}'
              f' loss_c:{avg_loss_c:0.2f} '
              f' time:{elapsed_time:0.2f}mins')

        if (epoch + 1) % 5 == 0:
            backbone_log_path = os.path.join(backbone_path_dir, f'{epoch+1}epoch.pth')
            hash_log_path = os.path.join(hash_path_dir, f'{epoch+1}epoch.pth')
            os.mknod(backbone_log_path)
            os.mknod(hash_log_path)
            torch.save(backbone, backbone_log_path)
            torch.save(hash_layer, hash_log_path)

        ####
        # temp = torch.transpose(long_features, 1, 2)  # (b,t,c)
        # temp = F.adaptive_avg_pool1d(temp, 1)  # (b,t,1)
        # temp = torch.squeeze(temp)  # (b,t)
        # temp = temp.to("cpu").detach().numpy()
        # writer.add_histogram("temporal features", temp, epoch)
        ####
        # writer.add_histogram("temporal weight",torch.squeeze(w).to("cpu").detach().numpy(),epoch)
        ####

    '''
    if (epoch + 1) % 10 == 0:
        db_binary, db_label = inference_db(db_loader, backbone,temporal_max_pool,hash_layer, hash_length, device)
        test_binary, test_label = inference_query(test_loader, backbone,hash_layer,hash_length, device)
        MAP_ = compute_MAP(db_binary, db_label, test_binary, test_label)
        print "MAP_: %s" % MAP_
        if MAP_ > MAP:
            MAP = MAP_
            named_model="max_backbone_e100_s40_MAX.pth"
            torch.save(backbone.state_dict(),"./models/"+named_model)
            torch.save(backbone.state_dict(),"./models/"+named_model)
            named_model="max_hash_layer_e100_s40_MAX.pth"
            torch.save(backbone.state_dict(),"./models/"+named_model)

    print "MAP: %s" % MAP
    f=open("MAP.log","a+")
    f.write(named_model+" "+str(MAP)+'\n\n')
    f.close()
    '''
    # torch.save(backbone.state_dict(), "./models/48_avg_b12_b120_backbone_e100_s40.pth")
    # # torch.save(temporal_attention.state_dict(),"./models/48_atten_quan_avg_b12_b120_temporal_attention_e100_s40.pth")
    # torch.save(hash_layer.state_dict(), "./models/48_avg_b12_b120_hash_layer_e100_s40.pth")

    # writer.close()
