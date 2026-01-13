from ros_api import ROSParamClient

client = ROSParamClient()
client.set_param('/estimated/scale', '1.5')