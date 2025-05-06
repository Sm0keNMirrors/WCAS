
import sys,os
import ssl
from urllib.request import build_opener
import datetime


def getDatesByTimes(start_day, end_day):
    result = []
    date_start = datetime.datetime.strptime(start_day, '%Y-%m-%d')
    date_end = datetime.datetime.strptime(end_day, '%Y-%m-%d')
    result.append(date_start.strftime('%Y-%m-%d'))
    while date_start < date_end:
        date_start += datetime.timedelta(days=1)
        result.append(date_start.strftime('%Y-%m-%d'))
    return result

if __name__ == "__main__":
# def downLoadFNLfiles(start_date, end_date, terminal):
    start_date = sys.argv[1] # 命令行给参数
    end_date = sys.argv[2]
    date_list = getDatesByTimes(start_date, end_date)
    hour_list = ['00', '06', '12', '18']
    filelist = []
    ssl._create_default_https_context = ssl._create_unverified_context  # grib1下载时，用于解决出现Linux端的SSL报错

    for date in date_list:
        year = date.split('-')[0]
        month = date.split('-')[1]
        day = date.split('-')[2]
        if int(year) <= 2007:  # FNL文件在2007年12月6日6时之前的都为grib1
            baseURL = 'https://data.rda.ucar.edu/ds083.2/grib1/'
            filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
            if month == '12' and day == '06':  # 2007年12月6日6时之前的都为grib1
                for hour in hour_list[0:2]:
                    filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib1'
                    filelist.append(filename)
                for hour in hour_list[2:4]:  # 2007年12月6日6时之后的都为grib2
                    baseURL = 'https://data.rda.ucar.edu/ds083.2/grib2/'
                    filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
                    filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib2'
                    filelist.append(filename)
            else:
                for hour in hour_list:  # 获取月份
                    filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib1'
                    filelist.append(filename)

        else:
            #baseURL = 'https://stratus.rda.ucar.edu/ds083.2/grib2/'
            baseURL = 'https://data.rda.ucar.edu/ds083.2/grib2/'
            filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
            for hour in hour_list:  # 获取月份
                filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib2'
                filelist.append(filename)

    opener = build_opener()
    for file in filelist:
        # 由于原先WCAS下载FNL的逻辑建立在'https://stratus.rda.ucar.edu/ds083.2/grib2/'下载指定时间段的fnl文件，
        # 时间段内有效的数据直接下载，无效的数据用空文件表示，空文件则说明GFS应以何时开始预报，而这个网站在2024.06.24左右失效，不再保留有效文件而全是空文件，则只能使用新url
        # 而新url在下载时，无效数据文件会下载失败，则为保留WCAS的执行逻辑不变，使用下述方法手动创建空文件
        if os.system(f'aria2c --file-allocation=none --check-certificate=false --dir=/home/WCAS_NCARfilesdownload/ {file}') == 0:  # 路径中有预设给服务器的下载文件存放路径
            pass
        else:
            tempfile = os.path.basename(file)
            os.system(f'touch /home/WCAS_NCARfilesdownload/{tempfile}')


    