import re, os, requests

# url = r"https://www.youtube.com/playlist?list=PLXO45tsB95cK7G-raBeTVjAoZHtJpiKh3"    #youtube播放列表
# headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
# html = requests.get(url, headers=headers).text
# # videoIds = re.findall('"videoId":"([A-Za-z0-9_-]{11})","thumbnail"', html)
# # for videoId in videoIds:
# #     print(videoId)
# #     download_link = "https://youtu.be/" + videoId  # 构造下载地址
# #     os.chdir(r"D:\DataSet数据集整理\CCV\videos")
# #     os.system("youtube-dl " + download_link)  # 用youtube-dl下载视频

# res = requests.get('https://www.ldoceonline.com/',headers={"User-Agent":"Mozilla/5.0"})
# download_link = "https://youtu.be/nymfb7yldZg"  # 构造下载地址
# os.chdir(r"D:\DataSet数据集整理\CCV\videos")
# os.system("youtube-dl " + download_link)  # 用youtube-dl下载视频
import time


def GetClassid(label_line):
    index=0
    for char in label_line:
        if char=='0':
            index+=1
        elif char=='1':
            return index
    return -1

#
# trainid_path=r'D:\DataSet数据集整理\CCV\trainVidID.txt'
# label_path=r'D:\DataSet数据集整理\CCV\trainLabel.txt'
# train_file=open(trainid_path,'r')
# label_file=open(label_path,'r')
# cnt=0
# class_limit=[]
# for i in range(20):
#     class_limit.append(150)  #其中只用到120个，30个是用来防止下载错误的
# print(class_limit)
# for (line,label_line )in zip(train_file,label_file):
#     # start=time.time()
#     # time.sleep(0)
#     # end = time.time()
#     # print('sleep '+str(end-start)+'s')
#     line = line[:-1]
#     label_line = label_line[:-1]
#     print(line,label_line)
#
#     classid=GetClassid(label_line)
#     print(classid)
#     class_limit[int(classid)]-=1
#     print(class_limit[int(classid)])
#     if(class_limit[int(classid)]<=0):
#         continue
#     try:
#         download_link = "https://youtu.be/"+line # 构造下载地址
#         # print("youtube-dl " + download_link + ' -o ' + str(cnt) + '_' + line+'.mp4')
#         print("youtube-dl " + download_link+' -o '+line+'.mp4')
#         download_path="D:/DataSet数据集整理/CCV/videos/trainset/"+str(classid)
#         if not os.path.exists(download_path):
#             os.mkdir(download_path)
#         os.chdir(download_path)
#         print('download_path=',download_path)
#         os.system("youtube-dl " + download_link+' -o '+line+'.mp4') #str(cnt)+'_'
#     except BaseException:
#         print('Error')
#     else:
#         print('Success')
#     cnt+=1



trainid_path=r'D:\DataSet数据集整理\CCV\testVidID.txt'
label_path=r'D:\DataSet数据集整理\CCV\testLabel.txt'
train_file=open(trainid_path,'r')
label_file=open(label_path,'r')
cnt=0
class_limit=[]
for i in range(20):
    class_limit.append(40)  #其中只用到25个，15个是用来防止下载错误的
print(class_limit)

for (line,label_line )in zip(train_file,label_file):
    start=time.time()
    time.sleep(5)
    end = time.time()
    # print('sleep '+str(end-start)+'s')
    line = line[:-1]
    label_line = label_line[:-1]
    print(line,label_line)

    classid=GetClassid(label_line)
    print(classid)
    if classid==-1:
        continue
    class_limit[int(classid)]-=1
    print(class_limit[int(classid)])
    if(class_limit[int(classid)]<=0):
        continue
    try:
        download_link = "https://youtu.be/"+line # 构造下载地址
        # print("youtube-dl " + download_link + ' -o ' + str(cnt) + '_' + line+'.mp4')
        print("youtube-dl " + download_link+' -o '+line+'.mp4')
        download_path="D:/DataSet数据集整理/CCV/videos/testset/"+str(classid)
        if not os.path.exists(download_path):
            os.mkdir(download_path)
        os.chdir(download_path)
        print('download_path=',download_path)
        os.system("youtube-dl " + download_link+' -o '+line+'.mp4') #str(cnt)+'_'
    except BaseException:
        print('Error')
    else:
        print('Success')
    cnt+=1



# testid_path=r'D:\DataSet数据集整理\CCV\testVidID.txt'
# test_file = open(testid_path, 'r')
# cnt=0
# for line in test_file:
#     download_link = "https://youtu.be/"+line # 构造下载地址
#     os.chdir(r"D:\DataSet数据集整理\CCV\videos\testset")
#     os.system("youtube-dl " + download_link+' -o '+str(cnt)+'_'+line)  # 用youtube-dl下载视频
#     cnt+=1

# download_link = "https://youtu.be/1DwOlDzZwW4"  # 构造下载地址
# os.chdir(r"D:")
# os.system("youtube-dl " + download_link)  # 用youtube-dl下载视频