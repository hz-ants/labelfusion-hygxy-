import sys
import os
import glob
from PIL import Image

focalLength = 616.29443359375
centerX = 315.2859903333336 
centerY = 244.88168334960938
scalingFactor = 1000

depths = [Image.open(depth_file) for depth_file in sorted(glob.glob('./depth/*.png'))]
rgbs =   [Image.open(rgb_file) for rgb_file in sorted(glob.glob('./rgb/*.png'))]

if len(depths) !=len(rgbs):
    raise Exception("number of color and depth pictures do not match")


for idx, item in enumerate(rgbs):
    points = []
    for v in range(item.size[1]):
        for u in range(item.size[0]):
            color = item.getpixel((u,v))
            Z = depths[idx].getpixel((u,v)) / scalingFactor
            if Z ==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open('./ply/{:04d}.ply'.format(idx), "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
element face 0
property list uchar int vertex_indices
end_header
%s
'''%(len(points),"".join(points)))
    file.close()