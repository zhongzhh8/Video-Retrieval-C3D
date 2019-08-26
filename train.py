import torch.utils.data as data
import torchvision.transforms as transforms

from data_loader import CustomDataset
from model import resnet18
from model import TemporalAvgPool
from model import HashLayer
from triplet_loss import TripletLoss
import time
import os
from utils import *




def load_data(root_folder, fpath_label, batch_size, shuffle=True, num_workers=16, long=False):
    transform = transforms.Compose([
        # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
        transforms.Resize((128, 171)),
        transforms.CenterCrop((112, 112)),  # Center
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])

    data_ = CustomDataset(root_folder=root_folder,
                          fpath_label=fpath_label,
                          transform=transform)

    # torch.utils.data.DataLoader
    loader_ = data.DataLoader(
        dataset=data_,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=shuffle,  # shuffle
        num_workers=num_workers)  # multi thread

    return loader_


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
    batch_size = 12  # 4*4
    num_epochs = 160  # 100
    step_size = 40  # 40
    learning_rate = 0.0001 #0.001
    hash_length = 48 #48
    margin = 14#14
    num_classes = 101  # 101
    load_model=False
    backbone_model_path='/home/disk3/a_zhongzhanhui/PycharmProject/video_retrieval_C3D/checkpoints/UCF101_64bits_20margin_101classes/backbone_0.772471.pth'
    hashlayer_model_path='/home/disk3/a_zhongzhanhui/PycharmProject/video_retrieval_C3D/checkpoints/UCF101_64bits_20margin_101classes/hashlayer_0.772471.pth'

    print('===start setting network and optimizer===')
    backbone = resnet18()
    load_state(backbone, "./pretrain/resnet-18-kinetics.pth") #加载保存好的模型
    backbone.to(device)
    backbone = torch.nn.DataParallel(backbone, device_ids=[0, 1, 2, 3])  # for multi gpu
    temporal_avg_pool = TemporalAvgPool().to(device) # temporal_max_pool = TemporalMaxPool().to(device)
    hash_layer = HashLayer(hash_length, num_classes).to(device)
    triplet_loss = TripletLoss(margin, device).to(device)

    params = list(backbone.parameters()) + list(hash_layer.parameters())
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size)
    print('===finish setting network and optimizer===')

    print('===setting data loader===')
    root_folder = "/home/disk3/a_zhongzhanhui/data/UCF-101/"
    train_fpath_label = "/home/disk3/a_zhongzhanhui/data/UCF-101/ucfTrainTestlist/train1.txt"
    test_fpath_label = "/home/disk3/a_zhongzhanhui/data/UCF-101/ucfTrainTestlist/test1.txt"

    train_loader = load_data(root_folder, train_fpath_label, int(batch_size * 10), True, num_workers=16)
    test_loader = load_data(root_folder, test_fpath_label, batch_size * 10, False, 8)
    db_loader = train_loader

    train_loader_iter = iter(cycle(train_loader)) #iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    print('===finish setting data loader===')

    backbone.train()
    hash_layer.train()

    checkpoint_path = './checkpoints/' + 'UCF101_' + str(hash_length) + 'bits_' + str(margin) + 'margin_' + str(
        num_classes) + 'classes'
    os.makedirs(checkpoint_path, exist_ok=True)

    if load_model:
        backbone.load_state_dict(torch.load(backbone_model_path))
        hash_layer.load_state_dict(torch.load(hashlayer_model_path))

        if isinstance(backbone, torch.nn.DataParallel):
            backbone = backbone.module
        if isinstance(hash_layer, torch.nn.DataParallel):
            hash_layer = hash_layer.module



    print('===start training===')
    maxMAP=0
    total_step = len(train_loader)  #batch数量
    for epoch in range(num_epochs):
        start_time = time.time()
        scheduler.step()
        epoch_loss = 0.
        for i in range(total_step): #逐个batch地遍历整个训练集
            # print('getting data')
            frames, labels = next(train_loader_iter)
            frames = frames.to(device)  # to gpu if possible
            labels = labels.to(device)  # to gpu if possible
            # print('getting feature')
            features = backbone(frames)  #shape=[94,512,2]
            features_avg = temporal_avg_pool(features)  #为毛要用平均池化
            h, c = hash_layer(features_avg)
            # print('calculating loss')
            loss = triplet_loss(h, labels)
            print(f'[epoch{epoch}-batch{i}] loss:{loss:0.4}')
            if loss == 0:
                continue
            ### Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / total_step
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'[{epoch}/{num_epochs}] loss:{avg_loss:0.5f} '
              f' time:{elapsed_time:0.2f} s')

        if epoch % 5 == 0:    #(epoch + 1) % 2 == 0:
            map_start_time=time.time()
            print('getting binary code and label')
            db_binary, db_label = inference(db_loader, backbone, temporal_avg_pool, hash_layer, hash_length, device)
            test_binary, test_label = inference(test_loader, backbone, temporal_avg_pool,hash_layer, hash_length, device)
            print('calculating mAP')
            MAP_ = compute_MAP(db_binary, db_label, test_binary, test_label)
            print("MAP_: %s" % MAP_)


            f = open(os.path.join(checkpoint_path, "MAP.log"), "a+")
            f.write('epoch:'+str(epoch) + "  loss:"+str(avg_loss) +'  mAP:'+ str(MAP_) + '\n')
            f.close()

            if MAP_ > maxMAP:
                maxMAP = MAP_
                backbone_pth_path = os.path.join(checkpoint_path, f'backbone_{MAP_:04f}.pth')
                hash_layer_pth_path = os.path.join(checkpoint_path, f'hashlayer_{MAP_:04f}.pth')
                torch.save(backbone.state_dict(), backbone_pth_path)
                torch.save(hash_layer.state_dict(), hash_layer_pth_path)

            map_end_time = time.time()
            print('calcualteing mAP used ', map_end_time - map_start_time, 's')



