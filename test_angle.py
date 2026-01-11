#!/usr/bin/env python3
from ros_api import JointAngleTCPClient

def main():
    # 1. 初始化客户端
    client = JointAngleTCPClient(host='localhost', port=51120, timeout=10)
    
    # 2. 示例1：设置关节角度
    print("=== 示例1：设置关节角度 ===")
    angle_list = [90.0 for _ in range(17)]  # 17个关节角度
    
    success, msg = client.set_joint_angles(angle_list, time_ms=200)
    if success:
        print(f"设置成功：{msg}")
    else:
        print(f"设置失败：{msg}")
    
    print("\n=== 示例2：获取关节角度 ===")
    # 3. 示例2：获取关节角度
    req_buf = ""
    success, result = client.get_joint_angles(req_type=0, req_buf=req_buf)
    if success:
        print(f"获取成功，角度数组长度：{len(result)}")
        print(f"角度数组：{result}")
    else:
        print(f"获取失败：{result}")

if __name__ == '__main__':
    main()