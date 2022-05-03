import glob
import numpy as np
import cv2
path = r'/content/Swin-Unet/data/Lung Segmentation/masks/*.png'
path2 = r'/content/Swin-Unet/data/Synapse/train_npz/'
# path = r'D:/github/Swin-Unet/*.png'
# path2=r'D:\github\Swin-Unet\CHNCXR_0007_0.png'
for i,img_path in enumerate(glob.glob(path)):
  image = cv2.imread(img_path)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
  print(str(img_path.split('/')[-1].split('.')[0]))
  label_path = img_path.replace('CXR_png','masks')
  label = cv2.imread(label_path,flags=0)
        #将非目标像素设置为0
  label[label!=255]=0
        #将目标像素设置为1
  label[label==255]=1
		#保存npz
  np.savez(path2+str(img_path.split('/')[-1].split('.')[0]),image=image,label=label)
  print('------------',i)
print('ok')