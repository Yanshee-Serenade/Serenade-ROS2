from ros_api import TrackingDataClient

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