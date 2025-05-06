import sys,os
import ssl
from urllib.request import build_opener
import urllib.request
import datetime
# import urllib3
import requests
import re
import time


def get_file_size(url: str, raise_error: bool = False) -> int:
        response = urllib.request.urlopen(url)
        file_size = response.headers['Content-Length']
        if file_size == None:
            if raise_error is True:
                raise ValueError('该文件不支持多线程分段下载！')
            return file_size
        return int(file_size)

if __name__ == "__main__":
# def downLoadGFSfiles(
#     save_path='.',
#     target_date_in = '2024-03-16-00:00',
#     after_hour = 8*3,
# ):
    """
    Athor: Xiang Dai
    """
    save_path = sys.argv[1] # 命令行给参数
    target_date_in = sys.argv[2]
    after_hour = int(sys.argv[3])
    # 本代码用于GFS每日自动下载，如需下载指定时间，需要在下面网页中确认是否存在该日期，一般为最近10天，历史存档仅保留0.25度，
    # 网站为rda.ucar.edu的ds084.1数据集

    os.system

    #http = urllib3.PoolManager(cert_reqs='CERT_NONE') # 服务器端报错解决

    # root_html = 'https://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/'   
    root_html = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'  # 使用NECP的http远程根目录，更加稳定且在无墙情形速度快
    save_path = save_path

    history_download = False
    target_date_download = True  # 如需下载单独时间，需要打开该标志     ↓这里格式没有空格，因为是以指令运行，多个空格会导致命令行中获取的参数不完整
    target_date = datetime.datetime.strptime(target_date_in, "%Y-%m-%d-%H:%M")  # 所需时间，UTC
    retry_times = 5
    after_hour = after_hour  # 往后预测的小时数，必须为3的倍数

    Session = requests.Session()
    page_1 = Session.get(root_html)
    obj1 = re.compile(r'>gfs.(?P<date_time>.*?)/</a>', re.S)
    info1 = obj1.finditer(page_1.text)

    page_1_date_list = list()
    for i1 in info1:
        date_time = (i1.group('date_time'))
        page_1_date_list.append(date_time)

    del page_1_date_list[-1]

    lose_gfs_date_list = list()
    if history_download:
        save_path_dir = os.listdir(save_path)

        for date_time_i in page_1_date_list:
            #if date_time_i not in save_path_dir:
            lose_gfs_date_list.append(date_time_i)
    else:
        lose_gfs_date_list.append(page_1_date_list[-1])

    day_hour = ['00', '06', '12', '18']
    if target_date_download:
        lose_gfs_date_list = [target_date.strftime('%Y%m%d')]
        day_hour = ['{0:02}'.format(target_date.hour)]

    for download_date in lose_gfs_date_list:
        save_date_path = save_path
        if not os.path.exists(save_date_path): os.mkdir(save_date_path)

        for hour_i in day_hour:
            save_date_hour_path = save_date_path
            if not os.path.exists(save_date_hour_path): os.mkdir(save_date_hour_path)

            for f_hour_i in range(after_hour//3):
                gfs_url = root_html + 'gfs.{0}/{1}/atmos/gfs.t{1}z.pgrb2.1p00.f{2:03}'.format(download_date, hour_i, f_hour_i*3)
                #if requests.get(gfs_url).status_code == 404:
                #    sys.exit()
                file_name = gfs_url.split('/')[-1]
                if not os.path.exists(save_date_hour_path+file_name):
                    # os.system('{0} --url {1} --path {2} --retry_times {3}'.format(download_exe_path, gfs_url, save_date_hour_path+file_name, retry_times))
                    os.system(f'aria2c --file-allocation=none --check-certificate=false --dir=/home/WCAS_NCARfilesdownload/ {gfs_url}')
                    time.sleep(1)

                url_file_size = get_file_size(gfs_url)
                if os.path.exists(save_date_hour_path+file_name) and (url_file_size != None) and (os.path.getsize(save_date_hour_path+file_name) != url_file_size):
                    # os.system('{0} --url {1} --path {2} --retry_times {3}'.format(download_exe_path, gfs_url, save_date_hour_path+file_name, retry_times))
                    os.system(f'aria2c --file-allocation=none --check-certificate=false --dir=/home/WCAS_NCARfilesdownload/ {gfs_url}')
                    time.sleep(1)
                if (os.path.getsize('/home/WCAS_NCARfilesdownload/'+save_date_hour_path+'/'+file_name) != url_file_size):
                    os.remove(save_date_hour_path+file_name)

        #     print('{0}  {1}:00起未来{2}小时预测数据下载完成!'.format(download_date, hour_i, after_hour))
        # print('{0} 当日gfs所有时间段未来{1}小时预测数据下载完成!'.format(download_date, after_hour))