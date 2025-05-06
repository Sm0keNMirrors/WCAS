import fcntl
import termios
import time
import paramiko
import os



if __name__ == "__main__":

    # 远程服务器的连接信息
    hostname = '94.74.121.127'
    port = 1120  # SSH默认端口
    username = 'root'
    password = 'Nh07#KpdO@Z@hGiW'

    # 本地文件路径和远程服务器上的目标路径
    local_path = '/data/WCAS/'
    remote_path = '/data/WCAS_NCARfilesdownload/'

    # 创建SSH客户端
    ssh = paramiko.SSHClient()
    # 自动添加不安全的SSH主机密钥
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接到远程服务器
    ssh.connect(hostname, port, username, password)

    file1 = "https://stratus.rda.ucar.edu/ds083.2/grib2/2023/2023.12/fnl_20231222_00_00.grib2"
    file1name = os.path.basename(file1)
    stdin, stdout, stderr = ssh.exec_command(f'aria2c --dir={remote_path} --file-allocation=none --check-certificate=false {file1}')
    print("文件远程下载完成。")

    # 使用SCP传输文件
    sftp = ssh.open_sftp()
    # sftp.put(local_path, remote_path)
    sftp.get(f"{remote_path}{file1name}", f"{local_path}{file1name}")

    # 关闭连接
    sftp.close()
    ssh.close()

    print("文件远程下载并获取完成。")


    