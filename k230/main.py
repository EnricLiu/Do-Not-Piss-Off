import sys

print("implementation:", sys.implementation)
print("platform:", sys.platform)
print("path:", sys.path)
print("Python version:", sys.version)

import json
import struct
from machine import UART
from machine import FPIOA

import os
from machine import LED

from media.sensor import * #导入camera模块，使用camera相关接口
from media.display import * #导入display模块，使用display相关接口
from media.media import * #导入media模块，使用meida相关接口
# from time import *

import nncase_runtime as nn #导入nn模块，使用nn相关接口
import ulab.numpy as np #导入np模块，使用np相关接口

import time
import math
import image

import gc

DISPLAY_WIDTH = ALIGN_UP(1920, 16)
DISPLAY_HEIGHT = 1080

OUT_RGB888P_WIDTH = ALIGN_UP(1024, 16)
OUT_RGB888P_HEIGH = 624

EMO_LABELS = ["happy", "surprised", "sad", "anger", "disgust", "fear", "neutral"]

confidence_threshold = 0.5
top_k = 5000
nms_threshold = 0.2
keep_top_k = 750
vis_thres = 0.5
variance = [0.1, 0.2]

anchors_path = '/sdcard/examples/18-NNCase/face_detection/prior_data_320.bin'
prior_data = np.fromfile(anchors_path, dtype=np.float)
prior_data = prior_data.reshape((4200,4))

scale = np.ones(4, dtype=np.uint8)*1024
scale1 = np.ones(10, dtype=np.uint8)*1024

g_kpu = None
ai2d = None
ai2d_builder = None

def free_kpu():
    global g_kpu
    if g_kpu is not None:
        del g_kpu
    g_kpu = nn.kpu()

def decode(loc, priors, variances):
    boxes = np.concatenate(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(scores,axis = 0)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        new_x1 = []
        new_x2 = []
        new_y1 = []
        new_y2 = []
        new_areas = []
        for order_i in order:
            new_x1.append(x1[order_i])
            new_x2.append(x2[order_i])
            new_y1.append(y1[order_i])
            new_y2.append(y2[order_i])
            new_areas.append(areas[order_i])
        new_x1 = np.array(new_x1)
        new_x2 = np.array(new_x2)
        new_y1 = np.array(new_y1)
        new_y2 = np.array(new_y2)
        xx1 = np.maximum(x1[i], new_x1)
        yy1 = np.maximum(y1[i], new_y1)
        xx2 = np.minimum(x2[i], new_x2)
        yy2 = np.minimum(y2[i], new_y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        new_areas = np.array(new_areas)
        ovr = inter / (areas[i] + new_areas - inter)
        new_order = []
        for ovr_i,ind in enumerate(ovr):
            if ind < thresh:
                new_order.append(order[ovr_i])
        order = np.array(new_order,dtype=np.uint8)
    return keep

def clip(val, start, end):
    if val < start:
        return start
    elif val > end:
        return end
    else:
        return val

def cubic_interpolation(t, a=-0.5):
    """
    Cubic interpolation kernel.
    t: Absolute distance from the center.
    a: Parameter for cubic interpolation. Default is -0.5 (Catmull-Rom spline).
    """
    # t = np.abs(t)
    if t <= 1 or t >= -1:
        return (a + 2) * t**3 - (a + 3) * t**2 + 1
    elif t < 2 or t > -2:
        return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
    else:
        return 0


def bicubic_resize(image, target):
    """
    Resize an image using BiCubic interpolation.
    image: Input image as a 2D (grayscale) or 3D (RGB) NumPy array.
    target_height: Desired height of the output image.
    target_width: Desired width of the output image.
    """

    target_height, target_width = target
    src_height, src_width = image.shape[-2:]
    scale_y = src_height / target_height
    scale_x = src_width / target_width

    target_shape = list(image.shape)
    target_shape[-1] = target_width
    target_shape[-2] = target_height

    output = np.zeros(target_shape, dtype=image.dtype)
    for y in range(target_height):
        for x in range(target_width):
            # Map target pixel (x, y) to source pixel coordinates
            src_x = x * scale_x
            src_y = y * scale_y

            # Find the integer part and fractional part
            x0 = math.floor(src_x)
            y0 = math.floor(src_y)
            dx = src_x - x0
            dy = src_y - y0

            # Perform bicubic interpolation
            patch = np.zeros(image[:, :, 0, 0].shape)  # Support for RGB or grayscale

            for m in range(-1, 3):
                for n in range(-1, 3):
                    # Calculate the weights
                    wx = cubic_interpolation(m - dx)
                    wy = cubic_interpolation(n - dy)

                    # Get source pixel value, clamping to image bounds
                    src_m = clip(y0 + n, 0, src_height - 1)
                    src_n = clip(x0 + m, 0, src_width - 1)
                    patch += wx * wy * image[:, :, src_m, src_n]

            # Assign the result to the output pixel
            output[:, :, y, x] = patch

    return output



def softmax(x):
    x = x[0]
    x_row_max = np.max(x,axis=-1)
    x_row_max = x_row_max.reshape(tuple(list(x.shape)[:-1]+[1]))
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = np.sum(x_exp,axis=-1).reshape(tuple(list(x.shape)[:-1]+[1]))
    softmax = x_exp / x_exp_row_sum

    return softmax

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def get_result(output_data):
    loc = []
    loc = np.zeros((1, 4200, 4), dtype=np.float)
    start_i = 0
    for _i in range(0, 3):
        sum_shape = 1
        for sh_i in output_data[_i].shape:
            sum_shape *= sh_i
        output_data[_i] = output_data[_i].reshape((1, -1, loc.shape[2]))
        loc[:,start_i:start_i + int(sum_shape/loc.shape[2]),:] = output_data[_i]
        start_i = start_i + int(sum_shape/loc.shape[2])

    #conf = []
    start_i = 0
    conf = np.zeros((1, 4200, 2), dtype=np.float)
    for _i in range(3, 6):
        sum_shape = 1
        for sh_i in output_data[_i].shape:
            sum_shape *= sh_i
        output_data[_i] = output_data[_i].reshape((1, -1, conf.shape[2]))
        conf[:,start_i:start_i + int(sum_shape/conf.shape[2]),:] = output_data[_i]
        start_i = start_i + int(sum_shape/conf.shape[2])
    conf = softmax(conf)

    boxes = decode(loc[0], prior_data, variance)
    boxes = boxes * scale
    scores = conf[:, 1]

    # ignore low scores
    inds = []
    boxes_ind = []
    scores_ind = []
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            inds.append(i)
            boxes_ind.append(boxes[i])
            scores_ind.append(scores[i])

    boxes_ind = np.array(boxes_ind)
    scores_ind = np.array(scores_ind)
    #landms = landms[inds]

    # keep top-K before NMS
    order = np.argsort(scores_ind, axis=0)[::-1][:top_k]
    boxes_order = []
    scores_order = []
    for order_i in order:
        boxes_order.append(boxes_ind[order_i])
        scores_order.append(scores_ind[order_i])
    if len(boxes_order)==0:
        return []
    boxes_order = np.array(boxes_order)
    scores_order = np.array(scores_order).reshape((-1,1))

    # do NMS
    dets = np.concatenate((boxes_order, scores_order), axis=1)
    keep = py_cpu_nms(dets, nms_threshold)

    dets_out = []
    for keep_i in keep:
        dets_out.append(dets[keep_i])
    dets_out = np.array(dets_out)

    # keep top-K faster NMS
    dets_out = dets_out[:keep_top_k, :]
    return dets_out


def emo_infer(frame, det):
    global g_kpu
    if frame is None or det is None: return None

    x1, y1, x2, y2 = map(lambda x: int(round(x)), det)
    if not (0 < x1 < x2 and 0 < y1 < y2):
        return None

    frame = frame.to_grayscale().to_numpy_ref()
#    frame = frame[:,y1:y2,x1:x2]
    frame = frame[y1:y2,x1:x2]
    frame = np.asarray([[frame, frame, frame]], dtype=np.uint8)
    height, width = y2 - y1, x2 - x1
#    size = max(height, width)
#    frame = frame[:,y1:y1+size,x1:x1+size]
    # print(frame.shape, height, width)
    # face = nn.from_numpy(frame)

    # with open(f"/data/images/{round(time.time())}.txt", "w", encoding="utf-8") as f:
    #     f.write(str(frame.tolist()))
    ####################### preprocessing #######################
    #           reshaping human face to (3, 128, 128)
    # free_ai2d()
    # global ai2d
    # ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.float)
    # ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    # kmodel_input = nn.from_numpy(np.ones((1,3,64,64),dtype=np.float))
    # transform = ai2d.build([1,3,height,width],[1,3,64,64])
    # transform.run(face, kmodel_input)
    # kmodel_input = kmodel_input.to_numpy()
    print("resizing....")
    kmodel_input = bicubic_resize(frame, (64, 64))
    kmodel_input = np.asarray(kmodel_input, dtype=np.float)

    ######################## inference ##########################
    print("inferring...")
    kmodel_input = (kmodel_input - 128) / 255
    kmodel_input = nn.from_numpy(kmodel_input)
    g_kpu.set_input_tensor(0, kmodel_input)
    g_kpu.run()

    results = []
    for i in range(g_kpu.outputs_size()):
        data = g_kpu.get_output_tensor(i)
        result = data.to_numpy()
        del data #tensor对象用完之后释放内存
        results.append(result)

#    del transform
    result = _softmax(results[0])[0]
    print(result)
    return list(zip(EMO_LABELS, result))

def face_detect(image):
    global ai2d_builder, g_kpu

    ai2d_input_tensor = nn.from_numpy(image)
    ai2d_out = nn.from_numpy(np.ones((1,3,320,320),dtype=np.uint8))
    ai2d_builder.run(ai2d_input_tensor, ai2d_out)

    g_kpu.set_input_tensor(0, ai2d_out)
    g_kpu.run()
    del ai2d_input_tensor, ai2d_out
    # get output
    results = []
    for i in range(g_kpu.outputs_size()):
        data = g_kpu.get_output_tensor(i)
        result = data.to_numpy()
        tmp = (result.shape[0],result.shape[1],result.shape[2],result.shape[3])
        result = result.reshape((result.shape[0]*result.shape[1],result.shape[2]*result.shape[3]))
        result = result.transpose()
        tmp2 = result.copy()
        tmp2 = tmp2.reshape((tmp[0],tmp[2],tmp[3],tmp[1]))
        del result
        results.append(tmp2)
    gc.collect()

    dets = get_result(results)

    if not dets: return None
    return sorted(dets, key=lambda x: x[-1])[-1][:4]

def inference(img_frame, osd_img):
    global g_kpu
    det = face_detect(img_frame.to_numpy_ref())
    gc.collect()

    # osd_img.clear()
    if det is None: return None

    # x1, y1, x2, y2 = map(lambda x: int(round(x, 0)), det[:4])
    # w = (x2 - x1) * DISPLAY_WIDTH // OUT_RGB888P_WIDTH
    # h = (y2 - y1) * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH
    # #绘制人脸框
    # osd_img.draw_rectangle(x1 * DISPLAY_WIDTH // OUT_RGB888P_WIDTH, y1 * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH, w, h, color=(255,255,0,255))
    # Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)

    print(det)
    free_kpu()
    g_kpu.load_kmodel("/data/kmodels/latest.kmodel")
    emo = emo_infer(img_frame, det)
    free_kpu()
    g_kpu.load_kmodel("/sdcard/examples/18-NNCase/face_detection/face_detection_320.kmodel")

    return emo

def blink_led(led_handler, duration_s=0.5, intensity=200):
    led_handler.value(intensity)
    led_handler.on()
    time.sleep(duration_s)
    led_handler.off()


def reduce(func: callable, iterable, initial=None):
    """
    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    """
    iterator = iter(iterable)
    try:
        result = next(iterator) if initial is None else initial
    except StopIteration:
        raise TypeError("reduce() of empty sequence with no initial value")

    for element in iterator:
        result = func(result, element)

    return result



class DataType():
    TOTAL_PHASE = 0x13
    BREATH_RATE = 0x14
    HEART_RATE  = 0x15
    DISTANCE    = 0x16

    def __init__(self, value):
        if value not in range(0x13, 0x17):
            raise ValueError

        self.value = value

    @staticmethod
    def from_byte(byte_value):
        try:
            return DataType(byte_value)
        except ValueError:
            raise ValueError(f"Invalid byte value for DataType: {byte_value}")

    def data_byte_num(self):
        return reduce(lambda x, y: x + y[0], self.structure(), 0)

    def structure(self):
        if self.value == self.TOTAL_PHASE:
            return [(4, 'f32', 'total_phase'), (4, 'f32', 'breath_phase'), (4, 'f32', 'heart_phase')]
        if self.value == self.BREATH_RATE:
            return [(4, 'f32', 'breath_rate')]
        if self.value == self.HEART_RATE:
            return [(4, 'f32', 'heart_rate')]
        if self.value == self.DISTANCE:
            return [(4, 'u32', 'flag'), (4, 'f32', 'distance')]

        raise ValueError(f"Invalid DataType: {self}")


def parse_radar_uart(read_chr: callable):
    def check_sum(data: bytes, sum: int):
        check = ~reduce(lambda x, y: x ^ y, data)
        return check + 256 == sum

    head = bytearray(7)

    # wait for the SOF
    while(True):
        chr = read_chr()
        # print(chr, chr == b'\x01')
        if chr == 0x01:
            idx = 0
            break

    # read the header
    head[0] = 0x01
    for idx in range(1, len(head)):
        chr = read_chr()
        head[idx] = chr

    # parse header
    sum = read_chr()
    if not check_sum(head, sum):
        raise ValueError("Invalid header checksum")
    frame_id = head[1] << 8 | head[2]
    data_len = head[3] << 8 | head[4]
    if head[5] != 0x0a:
        raise ValueError("Invalid frame type")
    data_type = DataType.from_byte(head[6])

    if data_len != data_type.data_byte_num():
        raise ValueError("Invalid data length")

    data = bytearray(data_len)
    while(data_len > 0):
        data_len -= 1
        data[data_len] = read_chr()

    data_sum = read_chr()
    if not check_sum(data, data_sum):
        raise ValueError("Invalid data checksum")

    idx = 0
    ret = {}
    structure = data_type.structure()
    for n, t, name in structure:
        if t == 'f32':
            d = struct.unpack('>f', data[idx:idx+n])[0]
        elif t == 'u32':
            d = struct.unpack('>I', data[idx:idx+n])[0]
        else:
            raise ValueError(f"Invalid data type: {t}")
        ret[name] = d

    return {
        'frame_id':     frame_id,
        'data_type':    data_type,
        'data':         ret
    }


fpioa = FPIOA()
fpioa.set_function(3, fpioa.UART1_TXD)
fpioa.set_function(4, fpioa.UART1_RXD)
#fpioa.set_function(3,set_ie=1,set_oe=1)
#fpioa.set_function(4,set_ie=1,set_oe=1)
fpioa.set_function(5, fpioa.UART2_TXD)
fpioa.set_function(6, fpioa.UART2_RXD)

uart_radar = UART(UART.UART1, baudrate=1382400, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
uart_pi = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
print(uart_radar, uart_pi)

def uart_radar_read_chr():
    global uart_radar
    while True:
        res = uart_radar.read(1)
        if res is not None and len(res) > 0:
            return res[0]


import _thread
data = {
    "total_phase":  None,
    "breath_phase": None,
    "heart_phase":  None,
    "breath_rate":  None,
    "heart_rate":   None,
    "emotion":      "neutral",
}
data_sem = _thread.allocate_lock()

sensor = Sensor()
sensor.reset()
sensor.set_hmirror(False)
sensor.set_vflip(False)
# 通道0直接给到显示VO，格式为YUV420
sensor.set_framesize(width = DISPLAY_WIDTH, height = DISPLAY_HEIGHT)
sensor.set_pixformat(Sensor.YUV420SP)
# 通道2给到AI做算法处理，格式为RGBP888
sensor.set_framesize(width = OUT_RGB888P_WIDTH , height = OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

# OSD图像初始化
# osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
osd_img = None

sensor_bind_info = sensor.bind_info(x = 0, y = 0, chn = CAM_CHN_ID_0)
Display.bind_layer(**sensor_bind_info, layer = Display.LAYER_VIDEO1)

# 设置为LT9611显示，默认1920x1080，
Display.init(Display.LT9611, to_ide = True)

ai2d = nn.ai2d()
ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
ai2d.set_pad_param(True, [0,0,0,0,0,125,0,0], 0, [104,117,123])
ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel )
ai2d_builder = ai2d.build([1,3,OUT_RGB888P_HEIGH,OUT_RGB888P_WIDTH], [1,3,320,320])

print("loading Kmodels...")
free_kpu()
g_kpu.load_kmodel("/sdcard/examples/18-NNCase/face_detection/face_detection_320.kmodel")
print("Kmodels Loaded!")
MediaManager.init()
# 启动sensor
sensor.run()

def ai_task():
    # media初始化
    rgb888p_img = None
    curr_emo = "neutral"

    while True:
        time.sleep(1)
        try:
            rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            if rgb888p_img == -1:
                print("face_detect_test, capture_image failed")
                time.sleep(1)
                continue

            if rgb888p_img.format() != image.RGBP888:
                time.sleep(1)
                continue

            emo = inference(rgb888p_img, osd_img)
            if emo is not None:
                emo = sorted(emo, key=lambda x: x[-1], reverse=True)
                emo = emo[0]
                if emo[1] > 0.5555:
                    curr_emo = emo[0]
                    data_sem.acquire()
                    data["emotion"] = curr_emo
                    data_sem.release()
                print(emo)
                time.sleep(5)
            else:
                pass
                # print("nothing at all :(")

            rgb888p_img = None

        except Exception as e:
            print(e)
            print("face_detect_test failed")

# finally:
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    #停止摄像头输出
    sensor.stop()
    #去初始化显示设备
    Display.deinit()
    #释放媒体缓冲区
    MediaManager.deinit()
    global ai2d, g_kpu
    del ai2d, g_kpu
    gc.collect()
    time.sleep(1)

    print("face_detect_test end")
    return 0

def update_radar_data_task():
    global data, data_sem
    while True:
        try:
            result = parse_radar_uart(uart_radar_read_chr)
            if result is not None:
                data_sem.acquire()
                for k, v in result["data"].items():
                    if k == 'flag': continue
                    if k == 'distance': continue
                    data[k] = v
                data_sem.release()
        except ValueError as e:
            if str(e) == "Invalid data length": continue
            pass
#            print(e)
        finally:
            time.sleep(0.004)

def send_radar_data_task():
    tmp_data = {}
    while True:
        data_sem.acquire()
        for k, v in data.items():
            tmp_data[k] = v
        data_sem.release()
        uart_pi.write(json.dumps(tmp_data) + "\n")
        print(tmp_data)
        time.sleep(1)

_thread.start_new_thread(update_radar_data_task, ())
_thread.start_new_thread(send_radar_data_task, ())
_thread.start_new_thread(ai_task, ())

while True:
    time.sleep(10)




#def reduce(func: callable, iterable):
#    """
#    Apply a function of two arguments cumulatively to the items of a sequence,
#    from left to right, so as to reduce the sequence to a single value.

#    Args:
#        func: A function that takes two arguments.
#        iterable: An iterable sequence.

#    Returns:
#        The reduced value.
#    """
#    it = iter(iterable)
#    value = next(it)
#    for element in it:
#        value = func(value, element)
#    return value


# class RadarMsg:
#     class DataType():
#         TOTAL_PHASE = 0x13
#         BREATH_RATE = 0x14
#         HEART_RATE  = 0x15
#         DISTANCE    = 0x16

#         def __init__(self, value):
#             if value not in range(0x13, 0x17):
#                 raise ValueError

#             self.value = value

#         @staticmethod
#         def from_byte(byte_value):
#             try:
#                 return RadarMsg.DataType(byte_value)
#             except ValueError:
#                 raise ValueError(f"Invalid byte value for DataType: {byte_value}")

#         def data_byte_num(self):
#             return reduce(lambda x, y: x + y[0], self.structure(), 0)

#         def structure(self):
#             if self.value == self.TOTAL_PHASE:
#                 return [(4, 'f32', 'total_phase'), (4, 'f32', 'breath_phase'), (4, 'f32', 'heart_phase')]
#             if self.value == self.BREATH_RATE:
#                 return [(4, 'f32', 'breath_rate')]
#             if self.value == self.HEART_RATE:
#                 return [(4, 'f32', 'heart_rate')]
#             if self.value == self.DISTANCE:
#                 return [(4, 'u32', 'flag'), (4, 'f32', 'distance')]

#             raise ValueError(f"Invalid DataType: {self}")

#     ID  : int
#     TYPE: DataType

#     def __init__(self, frame_id, data_type, raw_data):
#         self.ID = frame_id
#         self.TYPE = data_type
#         self._data = self.extract_params(data_type, raw_data)

#     def extract_params(self, type: DataType, raw_data: bytes):
#         if type == RadarMsg.DataType.TOTAL_PHASE:
#             return {"total_phase": raw_data }
#         if type == RadarMsg.DataType.BREATH_RATE:
#             return {"breath_rate": raw_data }
#         if type == RadarMsg.DataType.HEART_RATE:
#             return {"heart_rate": raw_data }
#         if type == RadarMsg.DataType.DISTANCE:
#             return {"distance": raw_data }

#     def __str__(self):
#         return "RadarMsg(ID={}, TYPE={}, DATA={})".format(self.ID, self.TYPE, self._data)

# def parse_radar_uart(iter):
#     def check_sum(data: bytes, sum: int):
#         return ~reduce(lambda x, y: x ^ y, data) + 256 == sum

#     idx = -1
#     head = bytearray(7)

#     # read the header
#     while(True):
#         char_data = next(iter)
#         if idx == -1:
#             if char_data != 0x01: continue
#         idx += 1
#         if idx >= 7: break
#         head[idx] = char_data

#     print(98)
#     # parse header
#     print("here")
#     if not check_sum(head, char_data):
#         raise ValueError("Invalid checksum {}, {}".format(head, char_data))
#     frame_id = head[1] << 8 | head[2]
#     data_len = head[3] << 8 | head[4]
#     if head[5] != 0x0a:
#         raise ValueError("Invalid frame type")
#     data_type = RadarMsg.DataType.from_byte(head[6])

#     data = bytearray(data_len)
#     while(data_len > 0):
#         data_len -= 1
#         data[data_len] = next(iter)

#     data_sum = next(iter)
#     if check_sum(data, data_sum):
#         return RadarMsg(frame_id, data_type, data)
#     raise ValueError("Invalid data checksum")
























