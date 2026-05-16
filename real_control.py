# -*- coding:utf-8 -*-
import math
import cv2
import numpy as np
import urx

import real_rtde
import ftsensor_read
import utils
import time

RAD2DEG = 180/math.pi

class UR5_Real(object):
    def __init__(self):
        self.RAD2DEG = 180/math.pi
        self.DEG2RAD = math.pi/180
        # self.rob = urx.Robot("192.168.1.21")
        self.rob = self._connect_robot("192.168.1.21")
        self.rob.set_tcp((0,0,0.3165,0,0,0))
        self.rob.set_payload(2.0, (0,0,0))
        self.rtde = real_rtde.UR5_Rtde()
        self.RTDE_SOFT_MODE = True
        self.RTDE_SOFT_F_THRESHOLD = 15
        self.RTDE_SOFT_RETURN = 0.0015
    
    def _connect_robot(self, ip_address, retry_delay=0.5):
        """
        尝试连接到机器人，直到连接成功。
        :param ip_address: 机器人 IP 地址
        :param retry_delay: 每次重试之间的延迟（秒）
        :return: 成功连接的 urx.Robot 对象
        """
        while True:
            try:
                print(f"Trying to connect to the robot at {ip_address}...")
                rob = urx.Robot(ip_address)  # 尝试连接
                print(f"Successfully connected to the robot at {ip_address}.")
                return rob  # 返回成功连接的机器人对象
            except Exception as e:
                print(f"Failed to connect to the robot: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # 等待后重试
        
    def __del__(self):
        self.rob.close()
        print("Robot Connection End")
        
    def close(self):
        self.rtde.close()
        self.rob.close()
        print("URX Robot Connection End")

    def getRealPosOrt_urx(self):
        posture = self.rob.getl()    # 获取位置和姿态
        # print("posture: ", posture)
        # 旋转矢量转欧拉角
        rotvector = np.array(posture[3:6]).reshape(3, 1)
        rot = cv2.Rodrigues(rotvector)[0]
        rpy = utils.rot2euler(rot).tolist()
        # return posture[0:3], rpy
        return np.concatenate([posture[0:3], rpy], axis=0)
    
    def getRealPosOrt(self):
        return self.rtde.getPosOrt()
    
        
    def getJointAngle(self):
        angle = self.rob.getj()
        return angle
    
    
    def setTCPPos(self, pos, acc=0.1, vel=0.2, wait=True):
        self.rob.movel(pos, acc=acc, vel=vel, wait=wait)
    
    def setPath(self, pos_via, pos_to):
        self.rob.movec(pos_via, pos_to)


    def setTCPPosOrt_urx(self, pos, ort, acc=0.1, vel=0.2, wait=True, relative=False):
        rot = utils.rpy_to_rotation(ort[0], ort[1], ort[2])
        vector= cv2.Rodrigues(rot)[0].tolist()
        rotvector = [i for item in vector for i in item]
        tcpPos = np.append(pos, np.array(rotvector))
        # print('tcpPos: ', tcpPos)
        self.rob.movel(tcpPos, acc=acc, vel=vel, wait=wait, relative=relative)
        self.rtde.refresh()

    def setTCPPosOrt(self, pos, ort, acc=0.25, vel=0.5, threshold_time=10, wait=True):
        pos_ort = np.concatenate([pos,ort], axis=0)
        "set ik movement of RTDE, simultaneously within the soft threshold"
        self.rtde.setPosOrt(pos_ort=pos_ort, speed=vel, acc=acc, threshold_time=threshold_time, isWait=wait, 
                            isSoft=self.RTDE_SOFT_MODE, soft_f_threshold=self.RTDE_SOFT_F_THRESHOLD, soft_return=self.RTDE_SOFT_RETURN)




    def moveFK_urx(self, joints_positions, isDEG=True,
               acc=0.2, vel=0.3, wait=True, relative=False, threshold=None): #使用角度制
        if isDEG:
            jp = [joints_positions[0]*self.DEG2RAD, joints_positions[1]*self.DEG2RAD, joints_positions[2]*self.DEG2RAD, joints_positions[3]*self.DEG2RAD, joints_positions[4]*self.DEG2RAD, joints_positions[5]*self.DEG2RAD]
        else:
            jp = [joints_positions[0], joints_positions[1], joints_positions[2], joints_positions[3], joints_positions[4], joints_positions[5]]
        joints = (jp[0], jp[1], jp[2], jp[3], jp[4], jp[5])
        self.rob.movej(joints, acc=acc, vel=vel, wait=wait, relative=relative, threshold=threshold)
        self.rtde.refresh()
        
    def moveFK(self, joints_positions, isDEG=True,
               acc=0.2, vel=0.3, wait=True, relative=False, threshold=None): #使用角度制
        if isDEG:
            jp = [joints_positions[0]*self.DEG2RAD, joints_positions[1]*self.DEG2RAD, joints_positions[2]*self.DEG2RAD, joints_positions[3]*self.DEG2RAD, joints_positions[4]*self.DEG2RAD, joints_positions[5]*self.DEG2RAD]
        else:
            jp = [joints_positions[0], joints_positions[1], joints_positions[2], joints_positions[3], joints_positions[4], joints_positions[5]]
        joints = (jp[0], jp[1], jp[2], jp[3], jp[4], jp[5])
        self.rtde.moveJ(joints_positions=joints_positions, speed=vel, acc=acc)
        
    
    def moveIK_urx(self, pos, ort, acc=0.25, vel=0.5, wait=True):
        self.setTCPPosOrt_urx(pos, ort, acc=acc, vel=vel, wait=wait)
        
    def moveIK(self, pos, ort, acc=0.25, vel=0.5, threshold_time=10, wait=True):
        self.setTCPPosOrt(pos, ort, acc=acc, vel=vel, threshold_time=threshold_time, wait=wait)
        # time.sleep(0.1)

    def moveIK_0(self, pos_ort, acc=0.25, vel=0.5, threshold_time=10, wait=True):
        self.moveIK(pos=pos_ort[0:3], ort=pos_ort[3:6], acc=acc, vel=vel, threshold_time=threshold_time, wait=wait)
    
    def moveIK_changePosOrt(self, d_pos, d_ort, acc=0.25, vel=0.5, threshold_time=10, wait=True):
        pos_ort = self.getRealPosOrt()
        # pos_ort = self.getRealPosOrt_urx()
        pos = np.array([pos_ort[0]+d_pos[0], pos_ort[1]+d_pos[1], pos_ort[2]+d_pos[2]])
        ort = np.array([pos_ort[3]+d_ort[0], pos_ort[4]+d_ort[1], pos_ort[5]+d_ort[2]])
        # print("pos: ", pos, "\nort: ", ort, "\nd_pos: ", d_pos, "\nd_ort: ", d_ort)
        self.moveIK(pos, ort, acc=acc, vel=vel, threshold_time=threshold_time, wait=wait)
        # self.moveIK_urx(pos, ort, acc=acc, vel=vel, wait=wait)
        
    def moveIK_changePosOrt_urx(self, d_pos, d_ort, acc=0.25, vel=0.5, threshold_time=10, wait=True):
        pos_ort = self.getRealPosOrt_urx()
        pos = np.array([pos_ort[0]+d_pos[0], pos_ort[1]+d_pos[1], pos_ort[2]+d_pos[2]])
        ort = np.array([pos_ort[3]+d_ort[0], pos_ort[4]+d_ort[1], pos_ort[5]+d_ort[2]])
        self.moveIK_urx(pos, ort, acc=acc, vel=vel, wait=wait)
    
    
    
    #相较于末端执行器运动的相关函数
    def moveIK_changePosOrt_onEnd(self, d_pos, d_ort, acc=0.25, vel=0.5, threshold_time=10, wait=True, is_urx=False):
        if is_urx is True:
            pos_ort = self.getRealPosOrt_urx()
        else:
            pos_ort = self.getRealPosOrt()
        nowTcp_mat = utils.PosOrt_to_HomogeneousMatrix(pos_ort)
        d_pos_ort = np.concatenate([d_pos, d_ort], axis=0)
        newTcp_to_nowTcp_mat = utils.PosOrt_to_HomogeneousMatrix(d_pos_ort)
        newTcp_mat = np.dot(nowTcp_mat, newTcp_to_nowTcp_mat)
        new_pos_ort = utils.HomogeneousMatrix_to_PosOrt(newTcp_mat)
        pos = new_pos_ort[0:3]; ort = new_pos_ort[3:6]
        if is_urx is True:
            self.moveIK_urx(pos, ort, acc=acc, vel=vel, wait=wait)
        else:
            self.moveIK(pos, ort, acc=acc, vel=vel, threshold_time=threshold_time, wait=wait)
            
    def get_d_pos_ort_onEnd(self, last_pos_ort, now_pos_ort):
        lastTcp_mat = utils.PosOrt_to_HomogeneousMatrix(last_pos_ort)
        nowTcp_mat = utils.PosOrt_to_HomogeneousMatrix(now_pos_ort)
        # lastTcp_mat_inv = utils.inverse_homogeneous_matrix_np(lastTcp_mat)
        lastTcp_mat_inv = utils.inverse_homogeneous_matrix_robotics(lastTcp_mat)
        nowTcp_to_lastTcp_mat = np.dot(lastTcp_mat_inv, nowTcp_mat)
        d_pos_ort = utils.HomogeneousMatrix_to_PosOrt(nowTcp_to_lastTcp_mat)
        d_pos_ort[3:6] = utils.normalize_angles(d_pos_ort[3:6])
        return d_pos_ort


    def openRG2_urx(self):
        self.rob.set_digital_out(0, False)

    def closeRG2_urx(self):
        self.rob.set_digital_out(0, True)
        
    def openRG2(self):
        # self.rtde.openRG2()
        self.openRG2_urx()
        self.rtde.refresh()

    def closeRG2(self):
        # self.rtde.closeRG2()
        self.closeRG2_urx()
        self.rtde.refresh()


    def readFTsensor(self):
        fx,fy,fz,tx,ty,tz = ftsensor_read.udp_get()
        return [fx,fy,fz,tx,ty,tz]


        


if __name__ == "__main__":
    real_robot = UR5_Real()
    # try:
        # for i in range(10):
        #     posture = real_robot.getPosOrient()
        #     print(posture)
    test_mode = 2
    if test_mode == 1:
        real_robot.openRG2()
        real_robot.closeRG2()
        real_robot.moveIK_urx(pos=[5.91e-3, -545.81e-3, 200e-3], ort=[3.1415, 0, -3.])
        real_robot.moveIK(pos=[5.91e-3, -645.81e-3, 200e-3], ort=[-3.1415, 0, -3.1415])
        real_robot.moveIK(pos=[5.91e-3, -545.81e-3, 200e-3], ort=[3.1415, 0, -3.])
        real_robot.moveIK_urx(pos=[5.91e-3, -645.81e-3, 200e-3], ort=[-3.1415, 0, -3.1415])
        real_robot.moveIK_urx(pos=[5.91e-3, -545.81e-3, 200e-3], ort=[-3.1415, 0, -3.1415])
        real_robot.moveIK(pos=[5.91e-3, -645.81e-3, 200e-3], ort=[3.1415, 0, -3.])
    elif test_mode == 2:
        rotvec = real_robot.rob.getl()
        print(rotvec)
    print("ok")
    real_robot.close()

    # finally:
    #     real_robot.rob.close()