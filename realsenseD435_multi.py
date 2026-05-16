import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time

class RealsenseD435Multi(object):

    # def __init__(self, width=640, height=480, frame_rate=15,
    #              exposure=312, gain=65, brightness=0, contrast=50, 
    #              gamma=160, hue=0, saturation=50, sharpness=100, white_balance=5500):
    def __init__(self, width=640, height=480, frame_rate=15,
                 exposure=24, gain=65, brightness=0, contrast=50, 
                 gamma=160, hue=0, saturation=50, sharpness=100, white_balance=5500):
        """初始化函数，支持多个相机连接并设置图像参数
        Args:
            width (int): 图像宽度，默认为640
            height (int): 图像高度，默认为480
            frame_rate (int): 帧率，默认为15
            exposure (int): 曝光值，默认为100
            gain (int): 增益值，默认为16
            brightness (int): 亮度值，默认为0
            contrast (int): 对比度值，默认为50
            gamma (int): 伽马值，默认为300
            hue (int): 色调值，默认为0
            saturation (int): 饱和度值，默认为64
            sharpness (int): 锐度值，默认为50
            white_balance (int): 白平衡值，默认为4600
        """
        self.im_width = width
        self.im_height = height
        self.frame_rate = frame_rate
        self.exposure = exposure
        self.gain = gain
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.hue = hue
        self.saturation = saturation
        self.sharpness = sharpness
        self.white_balance = white_balance

        self.pipelines = []  # 管道列表
        self.align = rs.align(rs.stream.color)  # 图像对齐对象
        self.intrinsics = []
        self.devices = []
        self.connect()

    def connect(self):
        """连接多个相机并初始化管道"""
        connect_device = []
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs.camera_info.serial_number))

        self.devices = connect_device  # 存储所有连接的设备

        # 启动每个相机的管道并设置参数
        for serial_number in self.devices:
            pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_device(serial_number)
            rs_config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.frame_rate)
            rs_config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.frame_rate)
            rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.frame_rate)
            rs_config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, self.frame_rate)
            profile = pipeline.start(rs_config)
            
            # 获取相机传感器并设置参数
            sensor = profile.get_device().query_sensors()[1]  # 获取彩色传感器
            sensor.set_option(rs.option.exposure, self.exposure)
            sensor.set_option(rs.option.gain, self.gain)
            sensor.set_option(rs.option.brightness, self.brightness)
            sensor.set_option(rs.option.contrast, self.contrast)
            sensor.set_option(rs.option.gamma, self.gamma)
            sensor.set_option(rs.option.hue, self.hue)
            sensor.set_option(rs.option.saturation, self.saturation)
            sensor.set_option(rs.option.sharpness, self.sharpness)
            sensor.set_option(rs.option.white_balance, self.white_balance)

            self.pipelines.append(pipeline)  # 将管道加入到列表


    def get_intrinsics(self, stream_index=0):
        """获取每个相机的内参"""
        profile = self.pipelines[stream_index].get_active_profile()
        rgb_stream = profile.get_stream(rs.stream.color)
        intrinsics = rgb_stream.as_video_stream_profile().get_intrinsics()
        return np.array([intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1]).reshape(3, 3)


    def get_data(self, sample_skip=0, get_w=None, get_h=None, offset_x=None, offset_y=None, is_show_window=True, is_wait_key=True):
        """获取对齐后的RGB图像和深度图像，支持多个相机，并可选择是否显示图像"""
        color_images = []
        depth_images = []
        gray_depth_images = []

        # 对每个相机的图像进行处理
        for i, pipeline in enumerate(self.pipelines):
            for _ in range(sample_skip):
                frames = pipeline.wait_for_frames()  # 丢弃一些初始化不稳定的帧
            frames = pipeline.wait_for_frames()  # 获取一帧
            aligned_frames = self.align.process(frames)  # 对齐

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # 获取RGB和深度图
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 对深度图进行灰度化处理
            min_depth = np.min(depth_image)
            max_depth = np.max(depth_image)
            gray_depth = np.uint8(255 * ((depth_image - min_depth) / (max_depth - min_depth)))
            gray_depth_image = np.stack((gray_depth,) * 3, axis=-1)

            # 可选裁剪
            if get_w is not None and get_h is not None:
                center_x, center_y = color_image.shape[1] // 2, color_image.shape[0] // 2
                start_x = max(center_x + offset_x[i] - get_w[i] // 2, 0)
                start_y = max(center_y + offset_y[i] - get_h[i] // 2, 0)
                end_x = min(start_x + get_w[i], color_image.shape[1])
                end_y = min(start_y + get_h[i], color_image.shape[0])
                color_image = color_image[start_y:end_y, start_x:end_x]
                depth_image = depth_image[start_y:end_y, start_x:end_x]
                gray_depth_image = gray_depth_image[start_y:end_y, start_x:end_x]

            color_images.append(color_image)
            depth_images.append(depth_image)
            gray_depth_images.append(gray_depth_image)

            # 显示窗口（如果需要）
            if is_show_window:
                window_name = f"Camera {i+1}"
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 设置窗口始终位于最前面
                imgs = np.hstack((color_image, gray_depth_image))
                cv2.imshow(window_name, imgs)
                # print(imgs)
            if is_wait_key: cv2.waitKey(1)
        return color_images, depth_images, gray_depth_images


    def sample_data_test(self, sample_skip=0, get_w=None, get_h=None, offset_x=None, offset_y=None):
        """实时获取和显示图像，按 's' 保存图像，按 'q' 退出"""
        save_path = os.path.join('Data', 'RealSense_test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saved_count = 0

        while True:
            color_images, depth_images, gray_depth_images = self.get_data(
            sample_skip=sample_skip, get_w=get_w, get_h=get_h, offset_x=offset_x, offset_y=offset_y, is_show_window=True,
            is_wait_key=False)

            # 键盘事件处理
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                saved_count += 1
                for idx, (color_image, depth_image, gray_depth_image) in enumerate(zip(color_images, depth_images, gray_depth_images)):
                    cv2.imwrite(os.path.join(save_path, "{:04d}_r_{}.png".format(saved_count, idx)), color_image)
                    cv2.imwrite(os.path.join(save_path, "{:04d}_d_{}.tiff".format(saved_count, idx)), depth_image)
                    cv2.imwrite(os.path.join(save_path, "{:04d}_d_gray_{}.png".format(saved_count, idx)), gray_depth_image)
                    print("{:04d}_r_{}.png 已保存！".format(saved_count, idx))
            elif key & 0xFF == ord('q') or key == 27:
                # 按下 'q' 或 'Esc' 键退出
                cv2.destroyAllWindows()
                break

    
    

    def stop(self):
        """停止所有相机的管道"""
        for pipeline in self.pipelines:
            pipeline.stop()


def test():
    camera = RealsenseD435Multi()
    save_path = os.path.join('Data', 'RealSense_test')
    "以2个相机为测试"
    # get_w = [640,640]; get_h=[480,480]
    # offset_x=[0,0]; offset_y=[0,0]
    get_w = [256,256]; get_h=[256,256]
    offset_x=[0,0]; offset_y=[0,25]
    for i in range(1):
        color_images, depth_images, gray_depth_images = camera.get_data(sample_skip=0,
                                                                        get_w=get_w, get_h=get_h, 
                                                                        offset_x=offset_x, offset_y=offset_y)
        for idx, color_image in enumerate(color_images):
            cv2.imwrite(os.path.join(save_path, "_test_{}.png".format(idx)), color_image)
        time.sleep(0.5)
    print("开始捕获图片，按s保存图像，按q退出")
    camera.sample_data_test(sample_skip=0,
                            get_w=get_w, get_h=get_h, 
                            offset_x=offset_x, offset_y=offset_y)
    camera.stop()


if __name__ == "__main__":
    test()
