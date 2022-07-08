import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import rawpy
import glob

class SeeInDark(nn.Module):
    def __init__(self, num_classes = 10):
        super(SeeInDark, self).__init__()

        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
        
    def lrelu(self,x):
        outx = torch.max(0.2 * x, x)
        return outx
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
#         conv1 = F.relu(self.conv1_1(x))
#         conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
#         conv2 = F.relu(self.conv2_1(pool1))
#         conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
#         conv3 = F.relu(self.conv3_1(pool2))
#         conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
#         conv4 = F.relu(self.conv4_1(pool3))
#         conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
#         conv5 = F.relu(self.conv5_1(pool4))
#         conv5 = F.relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
#         conv6 = F.relu(self.conv6_1(up6))
#         conv6 = F.relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
#         conv7 = F.relu(self.conv7_1(up7))
#         conv7 = F.relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
#         conv8 = F.relu(self.conv8_1(up8))
#         conv8 = F.relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
#         conv9 = F.relu(self.conv9_1(up9))
#         conv9 = F.relu(self.conv9_2(conv9))

        conv10= self.conv10_1(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out    
        
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512)

    im = np.expand_dims(im,axis=2) 
    H,W = im.shape[0:2]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()
    
if __name__=='__main__':

    model=SeeInDark().cuda()
    loss_function=nn.L1Loss().cuda()
    optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for idx in range(32):
        input_tensor=torch.zeros((32,4,256,256),dtype=torch.float32).cuda()
        gt_tensor=(torch.ones((32,3,512,512),dtype=torch.float32)).cuda()
        output=model(input_tensor)
        print(output.size())
#         if(idx == 1):
#           print(output)
        loss=loss_function(output,gt_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())