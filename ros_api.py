#!/usr/bin/env python3
import socket
import struct
import cv2
import numpy as np
import time
import datetime

class TrackingDataClient:
    def __init__(self, server_ip='127.0.0.1', port=51121):
        self.server_ip = server_ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 优化1：设置socket超时（避免无限阻塞）
        self.socket.settimeout(10.0)
        self.total_recv_size = 0  # 记录总接收字节数
    
    def connect_to_server(self):
        """连接到Python2 字节流服务器，添加连接调试日志"""
        try:
            print("=== [Client Init] Connecting to {}:{}... ===".format(self.server_ip, self.port))
            start_time = time.time()
            self.socket.connect((self.server_ip, self.port))
            cost_time = (time.time() - start_time) * 1000
            print("=== [Client Init] Connected successfully! Cost {:.2f} ms ===".format(cost_time))
            return True
        except socket.timeout:
            print("=== [Client Init] Connection timeout (10s) ===")
            return False
        except Exception as e:
            print("=== [Client Init] Connection failed: {} ===".format(e))
            return False
    
    def recv_all(self, length, desc="unknown data"):
        """确保接收指定长度的字节数据，添加接收调试日志，解决分包阻塞问题"""
        if length <= 0:
            print("=== [Recv Data] {}: Invalid length ({}), return empty ===".format(desc, length))
            return b''
        
        print("=== [Recv Data] {}: Waiting for {} bytes... ===".format(desc, length))
        data = b''
        start_time = time.time()
        while len(data) < length:
            try:
                remaining = length - len(data)
                chunk = self.socket.recv(min(remaining, 4096))  # 分块接收，每次最多4096字节
                if not chunk:
                    raise ConnectionAbortedError("Server closed the connection unexpectedly")
                data += chunk
                # 打印实时接收进度
                print("=== [Recv Data] {}: Received {}/{} bytes ({}%) ===".format(
                    desc, len(data), length, round(len(data)/length*100, 2)
                ))
            except socket.timeout:
                raise TimeoutError("Recv {} timeout (10s), received {}/{} bytes".format(desc, len(data), length))
        
        # 更新总接收字节数
        self.total_recv_size += len(data)
        cost_time = (time.time() - start_time) * 1000
        print("=== [Recv Data] {}: Received complete! {} bytes, cost {:.2f} ms ===".format(
            desc, len(data), cost_time
        ))
        return data
    
    def parse_point_cloud_data(self, pc_data, desc="unknown point cloud"):
        """解析点云原始字节数据为NumPy数组，添加解析调试日志"""
        print("=== [Parse Data] {}: Starting parse ({} bytes)... ===".format(desc, len(pc_data)))
        if not pc_data:
            print("=== [Parse Data] {}: Empty data, return empty array ===".format(desc))
            return np.array([])
        
        # 单个点的字节长度（3个float32，每个4字节）
        single_point_bytes = 12
        # 计算点云数量（总字节数 ÷ 单个点字节数）
        point_count = len(pc_data) // single_point_bytes
        if len(pc_data) % single_point_bytes != 0:
            print("=== [Parse Data] {}: Warning! Data size ({}) is not multiple of 12, discard remaining {} bytes ===".format(
                desc, len(pc_data), len(pc_data) % single_point_bytes
            ))
        
        # 初始化NumPy数组（N, 3），存储x/y/z
        point_cloud_np = np.zeros((point_count, 3), dtype=np.float32)
        
        # 循环解析每个点的x/y/z
        for i in range(point_count):
            # 计算当前点的起始字节偏移
            point_start_offset = i * single_point_bytes
            # 提取单个点的字节数据
            single_point_data = pc_data[point_start_offset:point_start_offset+single_point_bytes]
            
            # 解析x(0偏移)、y(4偏移)、z(8偏移)，均为float32（小端序）
            x = struct.unpack_from('<f', single_point_data, offset=0)[0]
            y = struct.unpack_from('<f', single_point_data, offset=4)[0]
            z = struct.unpack_from('<f', single_point_data, offset=8)[0]
            
            # 存入NumPy数组
            point_cloud_np[i] = [x, y, z]
        
        print("=== [Parse Data] {}: Parse completed! {} points, NumPy shape: {} ===".format(
            desc, point_count, point_cloud_np.shape
        ))
        return point_cloud_np
    
    def parse_byte_stream(self):
        """解析更新后的字节流，添加完整解析调试日志"""
        parsed_data = {}
        self.total_recv_size = 0  # 重置总接收字节数
        print("=== [Parse Stream] Starting byte stream parsing... ===")
        start_time = time.time()
        
        try:
            # 1. 解析内参（4个float64，32字节）
            intrinsics_data = self.recv_all(32, "Intrinsics")
            fx, fy, cx, cy = struct.unpack('>dddd', intrinsics_data)
            parsed_data['intrinsics'] = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
            print("=== [Parse Stream] 1. Intrinsics parsed: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f} ===".format(
                fx, fy, cx, cy
            ))
            
            # 2. 解析服务状态（1字节，uint8→bool）
            success_data = self.recv_all(1, "Service Success")
            success = struct.unpack('>B', success_data)[0] == 1
            parsed_data['success'] = success
            print("=== [Parse Stream] 2. Service status parsed: Success={} ===".format(success))
            
            # 3. 解析相机位姿（7个float64，56字节）
            pose_data = self.recv_all(56, "Camera Pose")
            x, y, z, qw, qx, qy, qz = struct.unpack('>ddddddd', pose_data)
            parsed_data['camera_pose'] = {
                'position': {'x': x, 'y': y, 'z': z},
                'orientation': {'w': qw, 'x': qx, 'y': qy, 'z': qz}
            }
            print("=== [Parse Stream] 3. Camera pose parsed: ===")
            print("  - Position (x,y,z): ({:.6f}, {:.6f}, {:.6f})".format(x, y, z))
            print("  - Orientation (w,x,y,z): ({:.6f}, {:.6f}, {:.6f}, {:.6f})".format(qw, qx, qy, qz))
            
            # 4. 解析相机坐标点云
            pc_camera_len_data = self.recv_all(4, "Camera Point Cloud Length")
            pc_camera_len = struct.unpack('>I', pc_camera_len_data)[0]
            parsed_data['point_cloud_camera_info'] = {'total_bytes': pc_camera_len}
            
            if pc_camera_len > 0:
                pc_camera_raw = self.recv_all(pc_camera_len, "Camera Point Cloud Data")
                point_cloud_camera_np = self.parse_point_cloud_data(pc_camera_raw, "Camera Point Cloud")
                parsed_data['tracked_points_camera'] = point_cloud_camera_np
                parsed_data['point_cloud_camera_info']['point_count'] = point_cloud_camera_np.shape[0]
            else:
                parsed_data['tracked_points_camera'] = np.array([])
                parsed_data['point_cloud_camera_info']['point_count'] = 0
            print("=== [Parse Stream] 4. Camera point cloud parsed: {} points ===".format(
                parsed_data['point_cloud_camera_info']['point_count']
            ))
            
            # 5. 解析世界坐标点云
            pc_world_len_data = self.recv_all(4, "World Point Cloud Length")
            pc_world_len = struct.unpack('>I', pc_world_len_data)[0]
            parsed_data['point_cloud_world_info'] = {'total_bytes': pc_world_len}
            
            if pc_world_len > 0:
                pc_world_raw = self.recv_all(pc_world_len, "World Point Cloud Data")
                point_cloud_world_np = self.parse_point_cloud_data(pc_world_raw, "World Point Cloud")
                parsed_data['tracked_points_world'] = point_cloud_world_np
                parsed_data['point_cloud_world_info']['point_count'] = point_cloud_world_np.shape[0]
            else:
                parsed_data['tracked_points_world'] = np.array([])
                parsed_data['point_cloud_world_info']['point_count'] = 0
            print("=== [Parse Stream] 5. World point cloud parsed: {} points ===".format(
                parsed_data['point_cloud_world_info']['point_count']
            ))
            
            # 6. 解析图像数据
            img_len_data = self.recv_all(4, "Image Length")
            img_len = struct.unpack('>I', img_len_data)[0]
            parsed_data['image_info'] = {'total_bytes': img_len}
            
            if img_len > 0:
                img_jpg_raw = self.recv_all(img_len, "Image Data")
                # 从字节流解码为NumPy数组，再转换为OpenCV图像
                nparr = np.frombuffer(img_jpg_raw, dtype=np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                parsed_data['current_image'] = cv_image
                parsed_data['image_info']['shape'] = cv_image.shape if cv_image is not None else None
            else:
                parsed_data['current_image'] = None
                parsed_data['image_info']['shape'] = None
            print("=== [Parse Stream] 6. Image parsed: {} bytes, shape: {} ===".format(
                img_len, parsed_data['image_info']['shape']
            ))
            
            # 打印解析总统计
            total_cost = (time.time() - start_time) * 1000
            print("=== [Parse Stream] Parsing completed! Total received: {} bytes, cost {:.2f} ms ===".format(
                self.total_recv_size, total_cost
            ))
            return parsed_data
        
        except TimeoutError as e:
            print("=== [Parse Stream] Timeout error: {} ===".format(e))
            return None
        except Exception as e:
            print("=== [Parse Stream] Parse failed: {} ===".format(e))
            return None
    
    def display_results(self, parsed_data):
        """展示解析结果，添加详细统计信息"""
        if not parsed_data:
            print("=== [Display] No valid data to display ===")
            return
        
        print("\n" + "="*60)
        print("=== Final Parsed Results ===")
        print("="*60)
        
        # 1. 打印内参
        print("\n--- Camera Intrinsics ---")
        intrinsics = parsed_data['intrinsics']
        for k, v in intrinsics.items():
            print("{}: {:.6f}".format(k, v))
        
        # 2. 打印服务状态
        print("\n--- Service Status ---")
        print("Success: {}".format(parsed_data['success']))
        
        # 3. 打印相机位姿
        print("\n--- Camera Pose ---")
        pose = parsed_data['camera_pose']
        pos = pose['position']
        ori = pose['orientation']
        print("Position (x,y,z): ({:.6f}, {:.6f}, {:.6f})".format(pos['x'], pos['y'], pos['z']))
        print("Orientation (w,x,y,z): ({:.6f}, {:.6f}, {:.6f}, {:.6f})".format(ori['w'], ori['x'], ori['y'], ori['z']))
        
        # 4. 打印相机坐标点云信息
        print("\n--- Point Cloud (Camera Coordinate) ---")
        pc_cam_info = parsed_data['point_cloud_camera_info']
        print("Total Bytes: {}, Point Count: {}".format(pc_cam_info['total_bytes'], pc_cam_info['point_count']))
        
        point_cloud_cam = parsed_data['tracked_points_camera']
        if point_cloud_cam.shape[0] > 0:
            print("Point Cloud NumPy Shape: {}".format(point_cloud_cam.shape))
            print("First 5 Points (x:像素x, y:像素y, z:相机深度):")
            print(point_cloud_cam[:5])
        
        # 5. 打印世界坐标点云信息
        print("\n--- Point Cloud (World Coordinate) ---")
        pc_world_info = parsed_data['point_cloud_world_info']
        print("Total Bytes: {}, Point Count: {}".format(pc_world_info['total_bytes'], pc_world_info['point_count']))
        
        point_cloud_world = parsed_data['tracked_points_world']
        if point_cloud_world.shape[0] > 0:
            print("Point Cloud NumPy Shape: {}".format(point_cloud_world.shape))
            print("First 5 Points (x:世界X, y:世界Y, z:世界Z):")
            print(point_cloud_world[:5])
        
            # 保留原有的图像信息打印部分
            print("\n--- Image Info ---")
            img_info = parsed_data['image_info']
            print("Total Bytes: {}, Image Shape: {}".format(img_info['total_bytes'], img_info['shape']))

            # 修改为保存图像到本地（替代原有的显示图像逻辑）
            cv_image = parsed_data['current_image']
            if cv_image is not None:
                # 生成带时间戳的保存路径
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                processed_path = f"image_{timestamp}.jpg"
                
                # 保存图像到本地
                save_success = cv2.imwrite(processed_path, cv_image)
                
                # 打印保存结果提示
                if save_success:
                    print(f"\n--- Image Saved Successfully ---")
                    print(f"Image saved to: {processed_path}")
                else:
                    print("\n--- Error ---")
                    print("Failed to save the image to local disk.")
            else:
                print("No valid image to display (and save)")
    
    def send_request(self):
        """优化：主动发送有效请求数据（解决服务器等待请求的阻塞问题）"""
        try:
            request_data = b"GET_TRACKING_DATA"  # 自定义简单请求标识
            print("=== [Client Request] Sending request: {} ({} bytes) ===".format(
                request_data, len(request_data)
            ))
            self.socket.sendall(request_data)
            return True
        except Exception as e:
            print("=== [Client Request] Failed to send request: {} ===".format(e))
            return False
    
    def close_connection(self):
        """关闭客户端连接，添加收尾日志"""
        self.socket.close()
        print("\n=== [Client] Connection closed ===".format())

if __name__ == '__main__':
    # 初始化客户端
    client = TrackingDataClient(server_ip='127.0.0.1', port=51121)
    
    # 连接服务器并解析数据
    if client.connect_to_server():
        # 关键优化：主动发送请求数据（解决服务器recv阻塞）
        client.send_request()
        parsed_data = client.parse_byte_stream()
        if parsed_data:
            client.display_results(parsed_data)
        client.close_connection()
    else:
        print("=== [Client] Exit due to connection failure ===")