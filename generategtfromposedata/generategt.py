import yaml 
import numpy as np 

meta_file = open('./posedata.yml', 'r')
meta = yaml.load(meta_file)
input_file = open('./test.txt')

test_lines = [] 
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] =='\n':
        input_line = input_line[:-1]
    test_lines.append(int(input_line))

#print(test_lines)


#for idx, item in enumerate(test_lines):
#    print('idx{0}, item{1}\n'.format(idx, item))

rs = [] # list of rotations
ts = [] # list of translations
poses = []
for idx, item in enumerate(test_lines):
    r = np.resize(np.array(meta[item][0]['cam_R_m2c']),(1,9))
    rs.append(r)
    t = np.resize(np.array(meta[item][0]['cam_t_m2c'])/1000, (1,3))
    ts.append(t)
    pose = np.concatenate((r,t), axis = 1)
    #r.shape()
    #t.shape()
    #pose = np.concatenate((r,t), axis = 1)
    #[r[0][0], r[0][1], r[0][2],r[0][3],r[0][4], r[0][5], r[0][6],  r[0][7], r[0][8],t[0][0],t[0][1], t[0][2]]
    poses.append(pose)

#print(rs[0].shape)
#print(r[1])
#print(r[2])
#print(ts[0].shape)
#print(poses[0])
#print(poses[0].shape)
#
# 
# 
# print(poses[0][0,0])
#print(poses[1])
#print(poses[2])
'''
#print(r[0])
#print(t[0])
#print(pose[0])
'''
#print(len(poses)) 1050 


lines =[] 
for i in range(len(poses)):
    lines.append('{:d} 3 {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.
    format(i, poses[i][0,0], poses[i][0,1],poses[i][0,2],poses[i][0,3],poses[i][0,4],poses[i][0,5],
          poses[i][0,6],poses[i][0,7],poses[i][0,8],poses[i][0,9],poses[i][0,10],poses[i][0,11]))

fo = open('gt.txt','w+')
fo.writelines(lines)
fo.close()
