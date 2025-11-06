# vision_test.py
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

class VisionTester:
    def __init__(self):
        # 初始化摄像头和MediaPipe组件
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查摄像头连接和权限。")
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 初始化检测模型
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 加载E字图片
        self.e_image = self._load_e_image('pictures/E.png')
        self._init_test_parameters()
        
    def _load_e_image(self, path):
        """加载并验证E字图片"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"E字图片文件未找到，请检查路径：{path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法加载E字图片，请检查路径：{path}")
        return img

    def _init_test_parameters(self):
        """初始化测试参数"""
        self.current_size = 0.5  # 初始大小
        self.current_direction = 0  # 0:右, 90:下, 180:左, 270:上
        self.min_size = 0.1
        self.max_size = 1.0
        self.correct_count = 0
        self.total_count = 0
        self.distance = 60  # 初始距离(厘米)
        self.consecutive_errors = 0  # 连续错误计数
        self.testing = True
        self.last_change_time = time.time()
        self.min_size_reached = False
        self.display_duration = 2.0  # E字显示时长
        self.start_time = time.time()  # 开始测试时间
        
        # 国际标准视力对照表
        self.VISION_STANDARD = {
            0.1: 4.0, 0.15: 4.2, 0.2: 4.4,
            0.25: 4.6, 0.3: 4.8, 0.4: 5.0,
            0.5: 5.2, 0.6: 5.4, 0.8: 5.6,
            1.0: 5.8
        }

    @staticmethod
    def rotate_image(img, angle):
        """旋转E字图片"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), 
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255, 0))

    def _detect_hand_direction(self, hand_landmarks):
        """改进后的手势方向检测"""
        # 获取关键点
        wrist = hand_landmarks.landmark[0]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # 计算向量和角度差
        vec_index = (index_tip.x - wrist.x, index_tip.y - wrist.y)
        vec_middle = (middle_tip.x - wrist.x, middle_tip.y - wrist.y)
        angle_diff = abs(math.degrees(math.atan2(vec_index[1], vec_index[0])) - math.degrees(math.atan2(vec_middle[1], vec_middle[0])))
        
        # 判断手指伸直状态
        if angle_diff < 20:
            angle = math.degrees(math.atan2(vec_index[1], vec_index[0]))
            return (int((angle + 45) % 360 // 90) * 90) % 360
        return None

    def _calculate_face_distance(self, face_landmarks):
        """基于面部特征计算距离"""
        left_eye = face_landmarks.landmark[159]
        right_eye = face_landmarks.landmark[386]
        eye_distance_px = math.hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y)
        
        # 使用更精确的物理模型计算距离
        average_face_width = 0.16  # 米
        focal_length = 600  # 相机焦距（估计值）
        distance = (average_face_width * focal_length) / (eye_distance_px * self.cap.get(3))
        return max(30, min(100, int(distance * 100)))  # 转换为厘米

    def _update_e_parameters(self):
        """根据检测结果更新E字参数"""
        # 调整大小
        if self.consecutive_errors > 0:
            self.current_size = min(self.max_size, self.current_size * 1.1)
        else:
            self.current_size = max(self.min_size, self.current_size * 0.9)
            self.min_size_reached = self.current_size == self.min_size
        
        # 改变方向
        self.current_direction = (self.current_direction + 90) % 360
        self.last_change_time = time.time()

    def run_test(self):
        """执行视力测试并返回最终结果"""
        try:
            print("请保持正对摄像头，正在校准距离...")
            self._calibrate_distance()
            print("\n开始视力检测，请保持正对摄像头...")
            print("手势说明：")
            print(" - 伸直手指指向方向回答E字方向")
            print(" - 连续两次错误回答将结束测试\n")
            
            while self.testing and self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    continue
                
                # 处理帧数据
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 面部距离检测
                face_results = self.face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    self.distance = self._calculate_face_distance(
                        face_results.multi_face_landmarks[0]
                    )
                
                # 手势方向检测
                hand_direction = None
                hand_results = self.hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    hand_direction = self._detect_hand_direction(
                        hand_results.multi_hand_landmarks[0]
                    )
                
                # 更新显示内容
                self._update_display(frame)
                
                # 检查是否需要检测手势
                if time.time() - self.last_change_time >= self.display_duration:
                    if hand_direction is not None:
                        self._process_direction_match(hand_direction)
                    
                    self._update_e_parameters()
                
                # 检查退出条件
                if self._check_exit_condition():
                    break
            
            final_vision = self._calculate_final_vision()
            print(f"\n最终视力结果: {final_vision:.1f}")
            return final_vision
            
        except Exception as e:
            print(f"测试异常：{str(e)}")
            return None
        finally:
            self._cleanup()

    def _process_direction_match(self, detected_direction):
        """处理方向匹配逻辑"""
        self.total_count += 1
        if detected_direction == self.current_direction:
            self.correct_count += 1
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1

    def _update_display(self, frame):
        """更新显示内容"""
        # 生成旋转后的E字
        rotated_e = self.rotate_image(self.e_image, self.current_direction)
        resized_e = cv2.resize(rotated_e, 
                              (int(rotated_e.shape[1] * self.current_size),
                               int(rotated_e.shape[0] * self.current_size)))
        
        # 合并到画面中央
        x_offset = (frame.shape[1] - resized_e.shape[1]) // 2
        y_offset = (frame.shape[0] - resized_e.shape[0]) // 2
        for c in range(3):
            frame[y_offset:y_offset+resized_e.shape[0], 
                 x_offset:x_offset+resized_e.shape[1], c] = \
                resized_e[:, :, c] * (resized_e[:, :, 3]/255.0) + \
                frame[y_offset:y_offset+resized_e.shape[0],
                     x_offset:x_offset+resized_e.shape[1], c] * (1.0 - resized_e[:, :, 3]/255.0)
        
        # 添加叠加信息
        info_lines = [
            f"视力: {self._calculate_vision_level():.1f}",
            f"距离: {self.distance}cm",
            f"准确率: {self.correct_count}/{self.total_count}",
            f"当前方向: {['右', '下', '左', '上'][self.current_direction // 90]}"
        ]
        
        for i, text in enumerate(info_lines):
            cv2.putText(frame, text, (10, 30 + 30*i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255) if i == 0 else (255, 255, 255), 2)
        
        cv2.imshow('Vision Test', frame)
        cv2.waitKey(5)

    def _calculate_vision_level(self):
        """计算当前视力水平"""
        closest_size = min(self.VISION_STANDARD.keys(),
                          key=lambda x: abs(x - self.current_size))
        return self.VISION_STANDARD[closest_size]

    def _calculate_final_vision(self):
        """计算最终视力结果"""
        return self._calculate_vision_level()

    def _check_exit_condition(self):
        """检查测试结束条件"""
        if self.consecutive_errors >= 2:
            print("\n测试结束 - 连续两次回答错误")
            self.testing = False
            cv2.destroyAllWindows()  # 关闭测试窗口
            return True
        return False

    def _cleanup(self):
        """释放资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.hands.close()
        print("资源已释放")
    
    def _calibrate_distance(self):
        """新增距离校准方法"""
        start_time = time.time()
        valid_samples = []
        
        while time.time() - start_time < 5:  # 5秒校准时间
            success, frame = self.cap.read()
            if not success:
                continue
            
            face_results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if face_results.multi_face_landmarks:
                distance = self._calculate_face_distance(face_results.multi_face_landmarks[0])
                valid_samples.append(distance)
                
            cv2.putText(frame, "正在校准距离...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration', frame)
            cv2.waitKey(1)
        
        if valid_samples:
            avg_distance = sum(valid_samples) / len(valid_samples)
            if abs(avg_distance - 60) > 10:  # 距离偏差超过10cm时提示
                print(f"检测到您当前距离为{avg_distance:.1f}cm，建议调整至60cm标准距离")
        cv2.destroyWindow('Calibration')

class VisionTestApp(VisionTester):
    def __init__(self, result_queue=None):
        super().__init__()
        self.result_queue = result_queue

    def run_test(self):
        """执行视力测试并返回最终结果"""
        try:
            result = super().run_test()
            if self.result_queue:
                self.result_queue.put(result)
            return result
        except Exception as e:
            print(f"测试异常：{str(e)}")
            if self.result_queue:
                self.result_queue.put(None)
            return None

if __name__ == "__main__":
    try:
        tester = VisionTester()
        result = tester.run_test()
        print(f"\n最终视力结果: {result:.1f}")
    except Exception as e:
        print(f"初始化失败: {str(e)}")