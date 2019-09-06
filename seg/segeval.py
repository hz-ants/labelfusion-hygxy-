from segnet import SegNet as segnet
import torch 
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import glob
import sys
sys.path.insert(0,'/home/fapsros/anaconda3/lib/python3.7/site-packages')
import cv2
model = segnet()
model.cuda()
model.load_state_dict(torch.load('./trained_models/stift_model_9_0.012945513147724607.pth'))
model.eval()

#colors = [np.array(Image.open(file).convert("RGB")) for file in sorted(glob.glob('/home/fapsros/Desktop/posedataset/dataset/stift/rgb/*.png')) ]
rgb = np.array(Image.open('./segmentation/rgb/3000.png'))

#colors_trans = [np.transpose(rgb, (2, 0, 1)) for rgb in colors]
rgb = np.transpose(rgb, (2, 0, 1))

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#colors_norm = [ norm(torch.from_numpy(rgb.astype(np.float32))) for rgb in colors_trans]
rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
#print(rgb)

#for idx, rgb in enumerate(colors_norm):
rgb = Variable(rgb).cuda()
semantic = model(rgb.unsqueeze(0))
_, pred_original = torch.max(semantic, dim=1)
pred = pred_original*255
img = np.transpose(pred.unsqueeze(0).cpu().numpy(), (1, 2, 0))
ret, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
torchvision.utils.save_image(pred, './3000.png')
#print(semantic)
#print(semantic.shape)
#print(pred)

