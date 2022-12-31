# 获取小车关节信息
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import os

p.connect(p.GUI)
p.setGravity(0, 0, -10)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # 隐藏侧边栏
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

p.resetSimulation()
p.setAdditionalSearchPath(pd.getDataPath())
planeId = p.loadURDF("3Dmodels/ground.SLDPRT/urdf/ground.SLDPRT.urdf")
base_path = os.getcwd()
car = p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[0,0,1], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
# while True:
#     p.stepSimulation()
#     time.sleep(1 / 240)
#     p.getCameraImage(320, 240)

# 得到机器人的关节数量
num_joints = p.getNumJoints(car)
joint_infos = []
for i in range(num_joints):
	joint_info = p.getJointInfo(car, i)
	print(joint_info)
	if joint_info[2] != p.JOINT_FIXED:
		if 'wheel' in str(joint_info[1]):
			joint_infos.append(joint_info)