import glob
import numpy as np
import cv2
path = r'/content/Swin-Unet/data/Lung Segmentation/*.png'
path_label = r'/content/Swin-Unet/data/Lung Segmentation/masks/'
path2 = r'/content/Swin-Unet/data/Synapse/train_npz/'
for i,img_path in enumerate(glob.glob(path)):
  image = cv2.imread(img_path)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
  label_path = path_label+str(img_path.split('/')[-1].split('.')[0])+'_mask.png'
  # label_path = label_path.replace('*.png','*_mask.png')
  # print(label_path)
  try:
    label = cv2.imread(label_path,flags=0)
          #将非目标像素设置为0
    label[label!=255]=0
          #将目标像素设置为1
    label[label==255]=1
      #保存npz
    np.savez(path2+str(img_path.split('/')[-1].split('.')[0]),image=image,label=label)
  except TypeError:
    print('mask 和 原图不匹配')
  else:
    print(str(img_path.split('/')[-1].split('.')[0])+'录入成功')
# print('ok')