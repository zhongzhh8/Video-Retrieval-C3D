import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
from data_loader import CustomDataset
from model import C3D_Hash_Model
from triplet_loss import TripletLoss
import time
import os
from utils import *




def load_data(root_folder, fpath_label, batch_size, shuffle=True, num_workers=16, train=False,num_frames=32):
    if train:
        transform = transforms.Compose([
            # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),  # Center
            transforms.RandomHorizontalFlip(),  # 训练集才需要做这一步处理，获得更多的随机化数据
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])
    else:
        transform = transforms.Compose([
            # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),  # Center
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])

    data_ = CustomDataset(root_folder=root_folder,
                          fpath_label=fpath_label,
                          transform=transform,)
                        #  num_frames=num_frames)

    # torch.utils.data.DataLoader
    loader_ = data.DataLoader(
        dataset=data_,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=shuffle,  # shuffle
        num_workers=num_workers)  # multi thread

    return loader_


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_parser():
    parser = argparse.ArgumentParser(description='train C3DHash')

    parser.add_argument('--dataset_name', default='HMDB', help='HMDB or UCF')
    parser.add_argument('--num_frames', type=int, default=32, help='number of frames taken form a video')
    parser.add_argument('--batch_size', type=int, default=120, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=160, help='number of epochs to train for')
    parser.add_argument('--step_lr', type=int, default=40, help='change lr per strp_lr epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='lr=0.001')
    parser.add_argument('--hash_length', type=int, default=48, help='length of hashing binary')
    parser.add_argument('--margin', type=float, default=14, help='取bit的四分之一多一点，margin影响很大')
    parser.add_argument('--load_model', default=False, help='wether load model checkpoints or not')
    parser.add_argument('--load_model_path',default='/home/disk3/a_zhongzhanhui/PycharmProject/video_retrieval_C3D/checkpoints/HMDB_48bits_14margin_/net_epoch50_mAP0.476344.pth',help='location to load model')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='checkpointing after batches')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # device configuration

    print('===start setting network and optimizer===')
    net = C3D_Hash_Model(opt.hash_length)
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=[2,3])  # for multi gpu

    if opt.load_model:
        net.load_state_dict(torch.load(opt.load_model_path))
        print('loaded model from '+opt.load_model_path)

    triplet_loss = TripletLoss(opt.margin, device).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_lr)
    print('===finish setting network and optimizer===')

    print('===setting data loader===')
    if opt.dataset_name=='UCF':
        root_folder = "/home/disk3/a_zhongzhanhui/data/UCF-101/"
        train_fpath_label = "/home/disk3/a_zhongzhanhui/data/UCF-101/TrainTestlist/train1.txt"
        test_fpath_label = "/home/disk3/a_zhongzhanhui/data/UCF-101/TrainTestlist/test1.txt"
    elif opt.dataset_name=='HMDB':
        root_folder = "/home/disk3/a_zhongzhanhui/data/HMDB-51/HMDB51/"
        train_fpath_label = "/home/disk3/a_zhongzhanhui/data/HMDB-51/TrainTestlist/labels/train1.txt"
        test_fpath_label = "/home/disk3/a_zhongzhanhui/data/HMDB-51/TrainTestlist/labels/test1.txt"
    train_loader = load_data(root_folder, train_fpath_label, opt.batch_size, shuffle=True, num_workers=16,) #train=False,num_frames=opt.num_frames
    test_loader = load_data(root_folder, test_fpath_label, opt.batch_size, shuffle= False, num_workers=8,) #train=False,num_frames=opt.num_frames
    db_loader = train_loader
    train_loader_iter = iter(cycle(train_loader)) #iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    print('===finish setting data loader===')


    checkpoint_path = './checkpoints/' + opt.dataset_name+'_' + str(opt.hash_length) + 'bits_' + str(opt.margin) + 'margin_' + str(opt.num_frames) + 'frames'
    os.makedirs(checkpoint_path, exist_ok=True)

    print('===start training===')
    maxMAP=0
    total_step = len(train_loader)  #batch数量
    for epoch in range(opt.num_epochs):
        net.train()
        start_time = time.time()
        scheduler.step()
        epoch_loss = 0.
        for i in range(total_step): #逐个batch地遍历整个训练集
            frames, labels = next(train_loader_iter)
            frames = frames.to(device)
            labels = labels.to(device)
            hash_features = net(frames)
            loss = triplet_loss(hash_features, labels)
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

        print(f'[{epoch}/{opt.num_epochs}] loss:{avg_loss:0.5f} '
              f' time:{elapsed_time:0.2f} s')

        if epoch % opt.checkpoint_step == 0:    #(epoch + 1) % 2 == 0:
            map_start_time=time.time()
            print('getting binary code and label')
            db_binary, db_label = inference(db_loader, net, opt.hash_length, device)
            test_binary, test_label = inference(test_loader, net, opt.hash_length, device)
            print('calculating mAP')
            MAP_ = compute_MAP(db_binary, db_label, test_binary, test_label)
            print("MAP_: %s" % MAP_)


            f = open(os.path.join(checkpoint_path, "MAP.log"), "a+")
            f.write('epoch:'+str(epoch) + "  loss:"+str(avg_loss) +'  mAP:'+ str(MAP_) + '\n')
            f.close()

            if MAP_ > maxMAP:
                maxMAP = MAP_
                save_pth_path = os.path.join(checkpoint_path, f'net_epoch{epoch}_mAP{MAP_:04f}.pth')
                torch.save(net.state_dict(), save_pth_path)

            map_end_time = time.time()
            print('calcualteing mAP used ', map_end_time - map_start_time, 's')



