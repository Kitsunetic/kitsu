import platform
import re
import socket
import subprocess


def get_system_info():
    s = {
        "cpu": "Unkown",
        "cpu_sockets": 1,
        "hostname": socket.gethostname(),
        "os_type": platform.system(),
        "gpu": "",
    }

    # CPU
    if s["os_type"] == "Linux":
        lscpu = subprocess.check_output("lscpu", shell=True).decode().strip().split("\n")
        for line in lscpu:
            try:
                if line.startswith("Model name:"):
                    s["cpu"] = " ".join(line.split()[2:])
                elif line.startswith("Socket(s):"):
                    s["cpu_sockets"] = int(line.split()[1])
            except:
                pass

    # GPU
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        gpus = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        s["gpu"] = gpus[0]
        if len(gpus) > 1:
            s["gpu"] += " x " + str(len(gpus))
    except subprocess.CalledProcessError:
        pass

    return s
