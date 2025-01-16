## @Author: ChatGPT
import wiringpi
import time
import threading

class Serial:
    """
    封装 wiringpi-python 的 serial 部分，使其 API 类似于 pyserial。
    """

    def __init__(self, port, baudrate=9600, timeout=None):
        """
        初始化串口。

        Args:
            port: 串口设备路径，例如 "/dev/ttyS0"。
            baudrate: 波特率，默认为 9600。
            timeout: 读取超时时间（秒），默认为 None（阻塞模式）。
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.fd = None  # 文件描述符
        self.lock = threading.Lock()  # 用于线程安全
        self.open()

    def open(self):
        """
        打开串口。
        """
        if self.fd is None:
            self.fd = wiringpi.serialOpen(self.port, self.baudrate)
            if self.fd < 0:
                raise IOError(f"Failed to open serial port: {self.port}")

    def close(self):
        """
        关闭串口。
        """
        if self.fd is not None:
            wiringpi.serialClose(self.fd)
            self.fd = None

    def write(self, data):
        """
        写入数据。

        Args:
            data: 要写入的字符串或字节数据。

        Returns:
            写入的字节数。
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        count = 0
        for byte in data:
            wiringpi.serialPutchar(self.fd, byte)
            count +=1
        return count

    def read(self, size=1):
        """
        读取指定数量的字节。

        Args:
            size: 要读取的字节数，默认为 1。

        Returns:
            读取到的字节数据 (bytes 类型)。
        """
        data = b''
        start_time = time.time()
        for _ in range(size):
            while wiringpi.serialDataAvail(self.fd) == 0:
                if self.timeout is not None and time.time() - start_time > self.timeout:
                    return data  # 超时
                time.sleep(0.001)  # 短暂休眠，避免 CPU 占用过高
            data += bytes([wiringpi.serialGetchar(self.fd)])
        return data

    def readuntil(self, eol='\n'):
        """
        读取一行数据，以指定字符结尾。

        Args:
            eol: 行结束符，默认为 '\n'。

        Returns:
            读取到的一行数据 (bytes 类型，包含 eol)。
        """
        line = b''
        start_time = time.time()
        while True:
            c = self.read(1)
            if not c:  # 读取到空数据，可能是超时或连接断开
                if self.timeout is not None and time.time() - start_time > self.timeout:
                    return line
                elif len(line) > 0: # 如果有读到数据，代表可能是连接断开
                    return line
                else:
                    continue

            line += c
            if c == eol.encode('utf-8'):
                return line

    def readline(self):
        return self.readuntil(eol='\n')[:-1]

    def in_waiting(self):
        """
        返回接收缓冲区中的字节数。

        Returns:
            接收缓冲区中的字节数。
        """
        return wiringpi.serialDataAvail(self.fd)

    def flush(self):
        """
        清空串口缓冲区。wiringpi没有相关方法，使用轮询实现
        """
        return wiringpi.serialFlush(self.fd)
    # 其他方法可以根据需要添加...