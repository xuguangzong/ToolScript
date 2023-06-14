import getpass
import psutil
import time

from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
from pathlib import Path
from typing import Optional


def kill_process(process_pid: int):
    """
    杀死进程
    :param process_pid:
    :return:
    """
    process = psutil.Process(process_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def get_available_space(path: str = "/") -> float:
    """
    获取硬盘可用空间， 单位M
    :param path:
    :return:
    """
    return psutil.disk_usage(path).free / 1024 / 1024


key = " This is the key of the dddddd "
# noinspection SpellCheckingInspection
iv = " ssssssssss "


def get_device_uuid() -> str:
    """
    获取设备UUID作为设备唯一标识
    :return:
    """
    with open("/sys/class/dmi/id/product_uuid", "r") as f:
        return f.read().rstrip()


def pad_text(text: str) -> bytes:
    """
    待加密数据补足长度为16的倍数
    :param text:
    :return:
    """
    pad_count = 16 - (len(text.encode("utf-8")) % 16)
    text = text + "\0" * pad_count
    return text.encode("utf-8")


def encrypt(text: str) -> str:
    """
    AES加密
    :param text:
    :return:
    """
    text = pad_text(text)
    cryptos = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    cipher_text = cryptos.encrypt(text)
    return b2a_hex(cipher_text).decode("utf-8")


def decrypt(text: str) -> Optional[str]:
    """
    解密
    :param text:
    :return:
    """
    cryptos = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    # noinspection PyBroadException
    try:
        plain_text = cryptos.decrypt(a2b_hex(text))
        return bytes.decode(plain_text).rstrip('\0')
    except Exception:
        return None


def generate_authorization_file(file: Path):
    """
    生成授权文件
    :param file:
    :return:
    """
    uuid = get_device_uuid()
    authorization_code = encrypt(uuid)
    input_key = getpass.getpass(f"设备编号：{uuid}，输入授权密钥:")
    if input_key == key:
        print(f"授权成功")
    elif input_key == authorization_code:
        print(f"授权成功")
    else:
        raise Exception(f"授权密钥错误")
    with open(file, "w") as f:
        f.write(authorization_code)


def verify_authorization(file: Path):
    """
    校验授权
    :param file: 要校验的授权文件，不存在或者校验失败时
    需要重新输入密钥生成授权文件，校验成功则放行
    :return:
    """
    if not file.exists():
        generate_authorization_file(file)
    with open(file, "r") as f:
        code = f.read().strip()
        uuid = get_device_uuid()
        authorization_code = encrypt(uuid)
        if code != authorization_code:
            raise Exception(f"密钥匹配错误，授权失败")


def generate_authorization(text: str):
    """
    生成授权密钥
    :param text: 机器号
    :return:
    """
    print(f"authorization is {encrypt(text)}")
    time.sleep(60)
