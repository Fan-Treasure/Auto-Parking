import os
import random
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

steering_angle = 0

class CustomEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, base_path=os.getcwd(), car_type='ackermann', mode='1', manual=False, multi_obs=False, render_video=False):
        """
        初始化环境

        :param render: 是否渲染GUI界面
        :param base_path: 项目路径
        :param car_type: 小车类型（ackermann）
        :param mode: 任务类型
        :param manual: 是否手动操作
        :param multi_obs: 是否使用多个observation
        :param render_video: 是否渲染视频
        """

        self.base_path = base_path
        self.car_type = car_type
        self.manual = manual
        self.multi_obs = multi_obs
        self.mode = mode
        assert self.mode in ['1', '2', '3', '4', '5', '6', '7', '8']  # mode不能超出1-6

        self.car = None
        self.done = False
        self.goal = None
        self.desired_goal = None

        self.ground = None
        self.left_wall1 = None
        self.right_wall1 = None
        self.front_wall1 = None
        self.left_wall2 = None
        self.right_wall2 = None
        self.front_wall2 = None
        self.left_wall3 = None
        self.right_wall3 = None
        self.front_wall3 = None
        self.left_wall4 = None
        self.right_wall4 = None
        self.front_wall4 = None
        self.left_wall5 = None
        self.right_wall5 = None
        self.front_wall5 = None
        self.left_wall6 = None
        self.right_wall6 = None
        self.front_wall6 = None
        self.parked_car1 = None
        self.parked_car2 = None
        self.parked_car3 = None

        # 定义状态空间
        obs_low = np.array([0, 0, -1, -1, -1, -1])
        obs_high = np.array([20, 20, 1, 1, 1, 1])
        if multi_obs:
            self.observation_space = spaces.Dict(
                spaces={
                    "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "achieved_goal": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "desired_goal": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                }
            )
        else:
            self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 4种动作：前进、后退、左转、右转

        # self.reward_weights = np.array([1, 0.3, 0, 0, 0.02, 0.02])
        if self.mode == '4':
            self.reward_weights = np.array([0.4, 0.9, 0, 0, 0.1, 0.1])
        elif self.mode == '5':
            self.reward_weights = np.array([0.65, 0.65, 0, 0, 0.1, 0.1])
        else:
            self.reward_weights = np.array([1, 1, 0, 0, 0.1, 0.1])
        self.reward_weights = np.array([1, 1, 0, 0, 0.1, 0.1])
        self.target_orientation = None
        self.start_orientation = None

        if self.mode in ['1', '2', '3', '4', '5', '6', '7', '8']:
            self.action_steps = 5
        else:
            self.action_steps = 3
        self.step_cnt = 0
        self.step_threshold = 2000

        if render:
            self.client = p.connect(p.GUI)  # 链接到物理引擎，并开启GUI显示
            time.sleep(1. / 240.)
        else:
            self.client = p.connect(p.DIRECT)  # 链接到物理引擎，不渲染GUI
            time.sleep(1. / 240.)
        if render and render_video:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 取消渲染？？
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 取消渲染时候周围的控制面板
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加资源路径，之后加载就不用绝对路径了
        p.setGravity(0, 0, -10)  # 设置重力

    def render(self, mode='human'):
        """
        渲染当前画面

        :param mode: 渲染模式
        """

        p.stepSimulation(self.client)
        time.sleep(1. / 240.)

    def reset(self):
        """
        重置环境

        """

        p.resetSimulation(self.client)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])
        p.setGravity(0, 0, -10)

        # 加载地面
        self.ground = p.loadURDF(os.path.join(self.base_path, "3Dmodels/ground.SLDPRT/urdf/ground.SLDPRT.urdf"), basePosition=[0, 0, 0.005], useFixedBase=10)

        p.addUserDebugLine([-0.88, -0.73, 0.02], [-0.88, 0.73, 0.02], [0.75, 0.75, 0.75], 3)  # 对坐标系添加辅助线，位置1，位置2，颜色，线宽
        p.addUserDebugLine([-0.88, -0.73, 0.02], [0.88, -0.73, 0.02], [0.75, 0.75, 0.75], 3)
        p.addUserDebugLine([0.88, 0.73, 0.02], [0.88, -0.73, 0.02], [0.75, 0.75, 0.75], 3)
        p.addUserDebugLine([0.88, 0.73, 0.02], [-0.88, 0.73, 0.02], [0.75, 0.75, 0.75], 3)

        # mode = 1, 2 (右下车位)
        if self.mode == '1' or self.mode == '2':
            self.left_wall1 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.45, 0.05, 0.01], useFixedBase=10)  # useFixedBase:强制加载的对象的基座是静止不动的
            self.right_wall1 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.73, 0.05, 0.01], useFixedBase=10)
            self.front_wall1 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_shuku.SLDPRT/urdf/front_boundary_shuku.SLDPRT.urdf"), basePosition=[0.46, -0.25, 0.01], useFixedBase=10)
        else:
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.45, 0.05, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.73, 0.05, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_shuku.SLDPRT/urdf/front_boundary_shuku.SLDPRT.urdf"), basePosition=[0.46, -0.25, 0.01], useFixedBase=10)
        #p.addUserDebugLine([0.45, 0.045, 0.01], [0.73, 0.045, 0.01], [0.98, 0.98, 0.98], 1)

        # mode = 3, 6 (中下)
        if self.mode == '3' or self.mode == '6':
            self.left_wall2 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.17, -0.10, 0.01], useFixedBase=10)
            self.right_wall2 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.45, 0.05, 0.01], useFixedBase=10)
            self.front_wall2 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_xieku.SLDPRT/urdf/front_boundary_xieku.SLDPRT.urdf"), basePosition=[0.17, -0.40, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
            self.parked_car1 = p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[0.06, -0.22, 0.05], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)  # getQuaternionFromEuler:欧拉角转换四元数
            self.parked_car2 = p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[-0.31, -0.347, 0.05], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2 + np.pi / 6]), useFixedBase=True)  # baseOrientation:在指定方向将对象的基础创建为世界空间四元数[X，Y， Z，W]
        else:
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.17, -0.10, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[0.45, 0.05, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_xieku.SLDPRT/urdf/front_boundary_xieku.SLDPRT.urdf"), basePosition=[0.17, -0.40, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
            p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[0.06, -0.22, 0.05], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)  # getQuaternionFromEuler:欧拉角转换四元数
            p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[-0.28, -0.35, 0.05], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2 + np.pi / 6]), useFixedBase=True)  # baseOrientation:在指定方向将对象的基础创建为世界空间四元数[X，Y， Z，W]
        #p.addUserDebugLine([0.17, -0.11, 0.01], [0.45, 0.04, 0.01], [0.98, 0.98, 0.98], 1)

        # mode = 4 (左下)
        if self.mode == '4':
            self.left_wall3 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.48, -0.28, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
            self.right_wall3 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.72, -0.42, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
            self.front_wall3 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_shuku.SLDPRT/urdf/front_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.56, -0.66, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
        else:
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.48, -0.28, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.72, -0.42, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_shuku.SLDPRT/urdf/front_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.56, -0.66, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
        #p.addUserDebugLine([-0.48, -0.29, 0.01], [-0.72, -0.43, 0.01], [0.98, 0.98, 0.98], 1)

        # mode = 5 (右上)
        if self.mode == '5':
            self.left_wall4 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[0.18, 0.71, 0.01], useFixedBase=10)
            self.right_wall4 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[0.58, 0.71, 0.01], useFixedBase=10)
            self.front_wall4 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_cefang.SLDPRT/urdf/front_boundary_cefang.SLDPRT.urdf"), basePosition=[0.18, 0.73, 0.01], useFixedBase=10)
            self.parked_car3 = p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[0.725, 0.57, 0.05], useFixedBase=True)
        else:
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[0.18, 0.71, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[0.58, 0.71, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_cefang.SLDPRT/urdf/front_boundary_cefang.SLDPRT.urdf"), basePosition=[0.18, 0.73, 0.01], useFixedBase=10)
            p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[0.725, 0.57, 0.05], useFixedBase=True)
        #p.addUserDebugLine([0.18, 0.475, 0.01], [0.58, 0.475, 0.01], [0.98, 0.98, 0.98], 1)

        # mode = 7 (中上)
        if self.mode == '7':
            self.left_wall5 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[0.04, 0.66, 0.01], useFixedBase=10)
            self.right_wall5 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[-0.36, 0.56, 0.01], useFixedBase=10)
            self.front_wall5 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_cefang.SLDPRT/urdf/front_boundary_cefang.SLDPRT.urdf"), basePosition=[-0.36, 0.58, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 12]), useFixedBase=10)
            self.parked_car4 = p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[-0.46, 0.436, 0.05], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi / 2]), useFixedBase=True)
        else:
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[0.04, 0.66, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_cefang.SLDPRT/urdf/side_boundary_cefang.SLDPRT.urdf"), basePosition=[-0.36, 0.56, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_cefang.SLDPRT/urdf/front_boundary_cefang.SLDPRT.urdf"), basePosition=[-0.36, 0.58, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 12]), useFixedBase=10)
            p.loadURDF("3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=[-0.46, 0.436, 0.05], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi / 2]), useFixedBase=True)
        #p.addUserDebugLine([-0.34, 0.32, 0.01], [0.04, 0.41, 0.01], [0.98, 0.98, 0.98], 1)

        # mode = 8 (左上)
        if self.mode == '8':
            self.left_wall6 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.88, 0.39, 0.01], useFixedBase=10)
            self.right_wall6 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.6, 0.56, 0.01], useFixedBase=10)
            self.front_wall6 = p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_xieku.SLDPRT/urdf/front_boundary_xieku.SLDPRT.urdf"), basePosition=[-0.88, 0.407, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
        else:
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.88, 0.39, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/side_boundary_shuku.SLDPRT/urdf/side_boundary_shuku.SLDPRT.urdf"), basePosition=[-0.6, 0.56, 0.01], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "3Dmodels/front_boundary_xieku.SLDPRT/urdf/front_boundary_xieku.SLDPRT.urdf"), basePosition=[-0.88, 0.407, 0.01], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]), useFixedBase=10)
        #p.addUserDebugLine([-0.86, 0.095, 0.01], [-0.6, 0.267, 0.01], [0.98, 0.98, 0.98], 1)

        basePosition = [0, 0, 0.05]
        if self.mode == '1':
            self.goal = np.array([0.6, -0.08])
            self.start_orientation = [0, 0, np.pi / 2]  # 初始姿态（欧拉角），即车的方向
            self.target_orientation = np.pi / 2
            basePosition = [0.6, 0.3, 0.05]
        elif self.mode == '2':
            self.goal = np.array([0.6, -0.08])
            self.start_orientation = [0, 0, np.pi]
            self.target_orientation = np.pi / 2
            basePosition = [0.35, 0.25, 0.05]
        elif self.mode == '3':
            self.goal = np.array([0.32, -0.175])
            self.start_orientation = [0, 0, np.pi / 2 + np.pi / 6]
            self.target_orientation = np.pi / 2
            basePosition = [0.2, 0.23, 0.05]
        elif self.mode == '4':
            self.goal = np.array([-0.518, -0.475])
            self.start_orientation = [0, 0, np.pi / 6]
            self.target_orientation = np.pi - np.pi / 3
            basePosition = [-0.4, 0.05, 0.05]
        elif self.mode == '5':
            self.goal = np.array([0.38, 0.61])
            self.start_orientation = [0, 0, np.pi / 3]
            self.target_orientation = 0
            basePosition = [0.18, 0.23, 0.05]
        elif self.mode == '6':
            self.goal = np.array([0.32, -0.175])
            self.start_orientation = [0, 0, np.random.rand() * 2 * np.pi]
            self.target_orientation = np.pi / 2
            while 1:
                random_x = (np.random.rand() - 0.5) * 3
                random_y = (np.random.rand() - 0.5) * 3
                if random_x < 0 and random_y > 0:
                    continue
                else:
                    break
            basePosition = [random_x, random_y, 0.05]
        elif self.mode == '7':
            self.goal = np.array([-0.15, 0.485])
            self.start_orientation = [0, 0, np.pi / 6]
            self.target_orientation = np.pi / 12
            basePosition = [-0.4, 0.03, 0.05]
        elif self.mode == '8':
            self.goal = np.array([-0.73, 0.327])
            self.start_orientation = [0, 0, np.pi / 6]
            self.target_orientation = - np.pi / 2
            basePosition = [-0.4, 0.03, 0.05]

        self.desired_goal = np.array([self.goal[0], self.goal[1], 0.0, 0.0, np.cos(self.target_orientation), np.sin(self.target_orientation)])

        # 加载小车
        self.t = Car(self.client, basePosition=basePosition, baseOrientationEuler=self.start_orientation, carType=self.car_type, action_steps=self.action_steps)
        self.car = self.t.car

        # 获取当前observation
        car_ob, self.vector = self.t.get_observation()
        observation = np.array(list(car_ob))

        self.step_cnt = 0

        if self.multi_obs:
            observation = {
                'observation': observation,
                'achieved_goal': observation,
                'desired_goal': self.desired_goal
            }

        return observation

    def distance_function(self, pos):
        """
        计算小车与目标点的距离（2-范数）

        :param pos: 小车当前坐标 [x, y, z]
        :return: 小车与目标点的距离
        """

        return np.sqrt(pow(pos[0] - self.goal[0], 2) + pow(pos[1] - self.goal[1], 2))

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        计算当前步的奖励

        :param achieved_goal: 小车当前位置 [x, y, z]
        :param desired_goal: 目标点 [x, y, z]
        :param info: 信息
        :return: 奖励
        """

        p_norm = 0.5
        reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.reward_weights)), p_norm)

        return reward

    def judge_collision(self):
        """
        判断小车与墙壁、停放着的小车是否碰撞

        :return: 是否碰撞
        """

        done = False
        if self.mode == '1' or self.mode == '2':
            points1 = p.getContactPoints(self.car, self.left_wall1)
            points2 = p.getContactPoints(self.car, self.right_wall1)
            points3 = p.getContactPoints(self.car, self.front_wall1)
        elif self.mode == '3' or self.mode == '6':
            points1 = p.getContactPoints(self.car, self.left_wall2)
            points2 = p.getContactPoints(self.car, self.right_wall2)
            points3 = p.getContactPoints(self.car, self.front_wall2)
        elif self.mode == '4':
            points1 = p.getContactPoints(self.car, self.left_wall3)
            points2 = p.getContactPoints(self.car, self.right_wall3)
            points3 = p.getContactPoints(self.car, self.front_wall3)
        elif self.mode == '5':
            points1 = p.getContactPoints(self.car, self.left_wall4)
            points2 = p.getContactPoints(self.car, self.right_wall4)
            points3 = p.getContactPoints(self.car, self.front_wall4)
        elif self.mode == '7':
            points1 = p.getContactPoints(self.car, self.left_wall5)
            points2 = p.getContactPoints(self.car, self.right_wall5)
            points3 = p.getContactPoints(self.car, self.front_wall5)
        elif self.mode == '8':
            points1 = p.getContactPoints(self.car, self.left_wall6)
            points2 = p.getContactPoints(self.car, self.right_wall6)
            points3 = p.getContactPoints(self.car, self.front_wall6)

        if len(points1) or len(points2) or len(points3):
            done = True
        if self.mode == '3' or self.mode == '6':
            points4 = p.getContactPoints(self.car, self.parked_car1)
            points5 = p.getContactPoints(self.car, self.parked_car2)
            if len(points4) or len(points5):
                done = True

        return done

    def step(self, action):
        """
        环境步进

        :param action: 小车动作
        :return: observation, reward, done, info
        """

        self.t.apply_action(action)  # 小车执行动作
        p.stepSimulation()
        car_ob, self.vector = self.t.get_observation()  # 获取小车状态

        position = np.array(car_ob[:2])
        distance = self.distance_function(position)
        reward = self.compute_reward(car_ob, self.desired_goal, None)

        if self.manual:
            print(f'dis: {distance}, reward: {reward}, center: {self.goal}, pos: {car_ob}')

        self.done = False
        self.success = False

        if distance < 0.02:
            reward = 1000
            self.success = True
            self.done = True

        self.step_cnt += 1
        if self.step_cnt > self.step_threshold:  # 限制episode长度为step_threshold
            self.done = True
        if car_ob[2] < -2:  # 小车掉出环境
            # print('done! out')
            reward = -1000
            self.done = True
        if self.judge_collision():  # 碰撞
            # print('done! collision')
            reward = -1000
            self.done = True
        if self.done:
            self.step_cnt = 0

        observation = np.array(list(car_ob))
        if self.multi_obs:
            observation = {
                'observation': observation,
                'achieved_goal': observation,
                'desired_goal': self.desired_goal
            }

        info = {'is_success': self.success}

        return observation, reward, self.done, info

    def seed(self, seed=None):
        """
        设置环境种子

        :param seed: 种子
        :return: [seed]
        """

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """
        关闭环境

        """

        p.disconnect(self.client)


class Car:
    def __init__(self, client, basePosition=[0, 0, 0.05], baseOrientationEuler=[0, 0, np.pi / 2],
                 max_velocity=6, max_force=100, carType='ackermann', action_steps=None):
        """
        初始化小车

        :param client: pybullet client
        :param basePosition: 小车初始位置
        :param baseOrientationEuler: 小车初始方向
        :param max_velocity: 最大速度
        :param max_force: 最大力
        :param carType: 小车类型
        :param action_steps: 动作步数
        """

        self.client = client
        urdfname = carType + '/' + carType + '.urdf'
        self.car = p.loadURDF(fileName="3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf", basePosition=basePosition, baseOrientation=p.getQuaternionFromEuler(baseOrientationEuler))

        self.steering_joints = [3, 7]
        self.drive_joints = [0, 1, 4, 8]

        self.max_velocity = max_velocity
        self.max_force = max_force
        self.action_steps = action_steps

    def apply_action(self, action):
        """
        小车执行动作

        :param action: 动作
        """
        global steering_angle
        velocity = self.max_velocity
        force = self.max_force
        streer = [3, 7]
        wheel = [4, 8]
        motor = [0, 1]
        for i in wheel:
            p.setJointMotorControl2(self.car, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        if action == 0:  # 前进
            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=velocity,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=-velocity,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.stepSimulation()
        elif action == 1:  # 后退
            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=-velocity,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=velocity,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.stepSimulation()


        elif action == 2:  # 左转
            if steering_angle > -np.pi / 4:  # 最大转向角
                steering_angle -= np.pi / 40

            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=0,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=0,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 3, p.POSITION_CONTROL,
                                        targetPosition= steering_angle)  # steering_angle是转向角
                p.setJointMotorControl2(self.car, 7, p.POSITION_CONTROL, targetPosition= -steering_angle)
                p.stepSimulation()
        elif action == 3:  # 右转
            if steering_angle < np.pi / 4:  # 最大转向角
                steering_angle += np.pi / 40

            for i in range(self.action_steps):
                p.setJointMotorControl2(self.car, 0, p.VELOCITY_CONTROL, targetVelocity=0,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 1, p.VELOCITY_CONTROL, targetVelocity=0,
                                        force=force)  # 驱动轮（后轮）v和f的定义在Car __init__
                p.setJointMotorControl2(self.car, 3, p.POSITION_CONTROL, targetPosition=steering_angle)  # steering_angle是转向角
                p.setJointMotorControl2(self.car, 7, p.POSITION_CONTROL, targetPosition=-steering_angle)
                p.stepSimulation()

    def get_observation(self):
        """
        获取小车当前状态

        :return: observation, vector
        """

        position, angle = p.getBasePositionAndOrientation(self.car)  # 获取小车位姿
        angle = p.getEulerFromQuaternion(angle)
        velocity = p.getBaseVelocity(self.car)[0]

        position = [position[0], position[1]]
        velocity = [velocity[0], velocity[1]]
        orientation = [np.cos(angle[2]), np.sin(angle[2])]
        vector = angle[2]

        observation = np.array(position + velocity + orientation)  # 拼接坐标、速度、角度

        return observation, vector
