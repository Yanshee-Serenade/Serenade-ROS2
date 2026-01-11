import YanAPI
import time

YanAPI.yan_api_init("raspberrypi")
print(time.time())
res = YanAPI.set_servos_angles({"NeckLR":90}, runtime=200)
print(time.time())
print(res)