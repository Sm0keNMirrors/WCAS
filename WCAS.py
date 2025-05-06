import fcntl
import shutil
import termios
import sys, os
import time
import subprocess
import ssl
import f90nml
import datetime
import logging
import wrf
from pyproj import CRS, Transformer
import multiprocessing
import math
import numpy as np
import netCDF4 as nc
import h5py
from osgeo import gdal,osr
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
from urllib.request import build_opener
import urllib.request
import urllib3
import requests
import re
import paramiko
from tqdm import tqdm



# @Author    :鲜耀涵;代翔

class Time():
    def Now(self):
        return str(datetime.datetime.now())

#计算函数
def add_matrices(data):
    m1, m2 = data
    return m1 + m2
def multiply_matrices(data):
    m1, m2 = data
    return m1 * m2
def parallel_compute_add(m1, m2,cores):
    # 创建进程池
    with multiprocessing.Pool(cores) as pool:
        # 映射add_matrices函数到所有数据上
        addition_result = pool.map(add_matrices, [(m1, m2)])
    return addition_result[0]
def parallel_compute_multipy(m1, m2,cores):
    # 创建进程池
    with multiprocessing.Pool(cores) as pool:
        # 映射multiply_matrices函数到所有数据上
        multiplication_result = pool.map(multiply_matrices, [(m1, m2)])
    return multiplication_result[0]
def day2doy(date):
    # 年月日转年积日
    # 输入列表 年月日 返回该日是年内第几日
    year = date[0]
    month = date[1]
    day = date[2]
    months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if 0 < month <= 12:
        sum = months[month - 1]
    else:
        print("month error")
    sum += day
    leap = 0
    # 接下来判断平年闰年
    # and的优先级大于or
    # 1、世纪闰年:能被400整除的为世纪闰年
    # 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
    if (year % 400 == 0) or ((year % 4) == 0) and (year % 100 != 0):
        leap = 1
    # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
    if (leap == 1) and (month > 2):
        sum += 1
    return sum
def getDatesByTimes(start_day, end_day):
    result = []
    date_start = datetime.datetime.strptime(start_day, '%Y-%m-%d')
    date_end = datetime.datetime.strptime(end_day, '%Y-%m-%d')
    result.append(date_start.strftime('%Y-%m-%d'))
    while date_start < date_end:
        date_start += datetime.timedelta(days=1)
        result.append(date_start.strftime('%Y-%m-%d'))
    return result
def getArrayVertices(ARR):
    top_left = ARR[0, 0]
    top_right = ARR[0, ARR.shape[1] - 1]
    bottom_left = ARR[ARR.shape[0] - 1, 0]
    bottom_right = ARR[ARR.shape[0] - 1, ARR.shape[1] - 1]
    return [top_left, top_right, bottom_left, bottom_right]
def getCbarmax(data):
    cbarmax = np.ceil(np.max(data) / 10) * 10
    cbarmin = np.floor(np.min(data) / 10) * 10
    if abs(np.max(data)) <= 10:
        cbarmax = np.ceil(np.max(data))
    if abs(np.min(data)) <= 10:
        cbarmin = np.floor(np.min(data))
    if cbarmin == 0 and cbarmax == 0: # 部分情况无值
        cbarmax,cbarmin = 1,-1
    return [cbarmax,cbarmin]
def wrf_grid_calculate(proj='lambert', lon_lat_limit=None, res_d01=36000, ratio=3, true_lat1=20, true_lat2=40,):
    """
    # Author: Dai xiang
    :param proj:
    :param lon_lat_limit:
    :param res_d01:
    :param ratio:
    :param true_lat1:
    :param true_lat2:
    :param wps_wrf_list:
    :param print_note:
    :return:
    """
    if not isinstance(lon_lat_limit, dict):
        print('lon_lat_limit不符合函数要求，重写填写后运行')
        exit()
    if proj not in ['lambert']:
        print('坐标系名称错误，请检查后运行')
        exit()

    max_dom = len(lon_lat_limit.keys())
    min_lon_list = list()
    max_lon_list = list()
    min_lat_list = list()
    max_lat_list = list()

    for dom_i in range(1, max_dom + 1):
        temp_dict = lon_lat_limit['d{0:02}'.format(dom_i)]
        min_lat_list.append(min(temp_dict[0]))
        max_lat_list.append(max(temp_dict[0]))
        min_lon_list.append(min(temp_dict[-1]))
        max_lon_list.append(max(temp_dict[-1]))

    geo_lon_list = list()
    geo_lat_list = list()
    for dom_i in range(max_dom):
        geo_col_num = math.ceil((max_lon_list[dom_i] - min_lon_list[dom_i]) / 0.01)
        geo_line_num = math.ceil((max_lat_list[dom_i] - min_lat_list[dom_i]) / 0.01)
        geo_lon_temp = np.empty([geo_line_num, geo_col_num])
        geo_lat_temp = np.empty([geo_line_num, geo_col_num])

        for geo_col_i in range(geo_col_num):
            geo_lon_temp[:, geo_col_i] = geo_col_i * 0.01 + min_lon_list[dom_i]
        for geo_line_i in range(geo_line_num):
            geo_lat_temp[geo_line_i, :] = max_lat_list[dom_i] - geo_line_i * 0.01
        geo_lon_list.append(geo_lon_temp)
        geo_lat_list.append(geo_lat_temp)

    ref_lon = (max_lon_list[0] + min_lon_list[0]) / 2
    ref_lat = (max_lat_list[0] + min_lat_list[0]) / 2
    stand_lon = (max_lon_list[0] + min_lon_list[0]) / 2

    e_we_list = list()
    e_sn_list = list()
    i_start_list = [1]
    j_start_list = [1]

    compute_code = 0

    if proj == 'lambert':
        crs_a = CRS.from_epsg(4326)
        crs_b = wrf.LambertConformal(TRUELAT1=true_lat1, TRUELAT2=true_lat2, MOAD_CEN_LAT=ref_lat,
                                     STAND_LON=ref_lon)
        crs_b = CRS.from_proj4(crs_b.proj4())
        transformer = Transformer.from_crs(crs_a, crs_b)

        lambert_grid_list = list()
        for dom_i in range(max_dom):
            lambert_grid_list.append(transformer.transform(geo_lat_list[dom_i], geo_lon_list[dom_i]))

        for dom_i in range(max_dom):
            if dom_i == 0:
                res_dv = res_d01
                col_d = math.ceil(
                    (np.max(lambert_grid_list[dom_i][0]) - np.min(lambert_grid_list[dom_i][0])) / res_dv)
                line_d = math.ceil(
                    (np.max(lambert_grid_list[dom_i][1]) - np.min(lambert_grid_list[dom_i][1])) / res_dv)

            if dom_i > 0:
                res_dv = res_d01
                if dom_i > 1:
                    res_dv = res_d01 / (ratio ** (dom_i - 1))

                col_start = math.ceil(
                    (np.min(lambert_grid_list[dom_i][0]) - np.min(lambert_grid_list[dom_i - 1][0])) / res_dv)
                line_start = math.ceil(
                    (np.min(lambert_grid_list[dom_i][1]) - np.min(lambert_grid_list[dom_i - 1][1])) / res_dv)
                col_end = math.ceil(
                    (np.max(lambert_grid_list[dom_i][0]) - np.min(lambert_grid_list[dom_i - 1][0])) / res_dv)
                line_end = math.ceil(
                    (np.max(lambert_grid_list[dom_i][1]) - np.min(lambert_grid_list[dom_i - 1][1])) / res_dv)
                col_d = (col_end - col_start) * ratio + 1
                line_d = (line_end - line_start) * ratio + 1

                i_start_list.append(col_start)
                j_start_list.append(line_start)

            e_we_list.append(col_d)
            e_sn_list.append(line_d)

        compute_code = 1
    grid_dict = {}
    grid_dict.update({'e_we':e_we_list})
    grid_dict.update({'e_sn': e_sn_list})
    grid_dict.update({'i_start': i_start_list})
    grid_dict.update({'j_start': j_start_list})
    grid_dict.update({'ref_lon': round(ref_lon, 2)})
    grid_dict.update({'ref_lat': round(ref_lat, 2)})
    grid_dict.update({'truelat1': true_lat1})
    grid_dict.update({'truelat2': true_lat2})
    grid_dict.update({'stand_lon': round(stand_lon, 2)})

    return grid_dict
def get_file_size(url: str, raise_error: bool = False) -> int:
        response = urllib.request.urlopen(url)
        file_size = response.headers['Content-Length']
        if file_size == None:
            if raise_error is True:
                raise ValueError('该文件不支持多线程分段下载！')
            return file_size
        return int(file_size)

#绘图函数
def drawCoutourf(
    dataset, 
    londata, 
    latdata, 
    shp_dirlist, 
    title, 
    picout_dir, 
    colorbarcmap, 
    extent,
    colorbarinterval,
    FileType,
    winddatas='',
    barmin = 0,
    barmax = 0,
    barlabel = '',
    labelsize=24,
    windscale=15,
    windwidth=0.0023
):
    font_path = f"{WCAS_dir}input/fonts/times+simsun.ttf"
    matplotlib.font_manager.fontManager.addfont(font_path)
    prop = matplotlib.font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    matplotlib.rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    matplotlib.rcParams['font.size'] = 13  # 设置字体大小
    matplotlib.rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
    # matplotlib.rcParams['font.size'] = 13  # 设置字体大小
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(6, 10),dpi=200)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    if (FileType == 'WRF'):  # 两种模式输出结果数据结构不同，WRF：dataset[:][0] CMAQ:dataset[:][0][0]
        data_cf = ax.contourf(londata, latdata, dataset[:][0], cmap=colorbarcmap,
                              levels=np.linspace(barmin, barmax, 80 + 1))  # 必须+1才能取得整端点cbar
        if winddatas != '':
            ax.quiver(winddatas[0][0], winddatas[1][0], winddatas[2][0], winddatas[3][0], transform=proj, 
                    #   scale=np.max(np.sqrt(winddatas[2][0]**2 + winddatas[3][0]**2)) * 15, # 通过最大值来确定风的sacle
                      scale=windscale,
                    scale_units='inches', width=0.0022)
    if (FileType == 'WRF_hourly'):
        data_cf = ax.contourf(londata, latdata, dataset[:], cmap=colorbarcmap,
                              levels=np.linspace(barmin, barmax, 80 + 1))  
        if winddatas != '':  #
            ax.quiver(winddatas[0], winddatas[1], winddatas[2], winddatas[3], transform=proj, 
                      scale=windscale,
                  scale_units='inches', width=windwidth)
    if (FileType == 'CMAQ'):
        data_cf = ax.contourf(londata, latdata, dataset[:][0], cmap=colorbarcmap,
                              levels=np.linspace(barmin, barmax, 80 + 1)) 
        if winddatas != '':
            ax.quiver(winddatas[0][0], winddatas[1][0], winddatas[2][0], winddatas[3][0], transform=proj, 
                      scale=windscale,
                    scale_units='inches', width=0.0022)
    if (FileType == 'CMAQ_hourly'):
        data_cf = ax.contourf(londata, latdata, dataset[:], cmap=colorbarcmap,
                            levels=np.linspace(barmin, barmax, 80 + 1)) 
        if winddatas != '':  
            ax.quiver(winddatas[0], winddatas[1], winddatas[2], winddatas[3], transform=proj, 
                      scale=windscale,
                  scale_units='inches', width=windwidth)

    for shp_dir in shp_dirlist:
        shp = cfeat.ShapelyFeature(Reader(shp_dir).geometries(), proj, edgecolor='k', facecolor='none')
        ax.add_feature(shp, lw=0.5, zorder=2)
    ax.set_extent(extent)  # 显示范围123
    position = fig.add_axes([0.1, 0.15, 0.8, 0.03])  # colorbar位置和大小 竖向
    cb = fig.colorbar(data_cf, cax=position, orientation='horizontal', extend='both',
                      ticks=np.linspace(barmin, barmax, round(barmax / colorbarinterval) + 1),
                      # 等间隔的bar分隔值colorbarinterval,计算出间隔数
                      format='%i', fraction=0.046, pad=0.04,shrink=0.8)  # orientation='horizontal' 
    # fig.subplots_adjust(up=0.8) # 调整主轴，使得Cbar不会重叠
    cb.ax.tick_params(labelsize=labelsize*0.5)  # 刻度字体大小 竖向
    cb.set_label(label=barlabel)  # 设置colorbar的标签字体及其大小
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.2, linestyle='--')
    # gl.xlabel_style = {'size': np.floor(labelsize*0.5)}  # 设置经度标签字体大小
    # gl.ylabel_style = {'size': np.floor(labelsize*0.5)}  # 设置纬度标签字体大小
    ax.set_title(title)# fontdict={'size': labelsize*0.75}
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    plt.savefig(picout_dir + title+'.png')
    matplotlib.pyplot.close()
def drawGeogrid(   # 绘制geogrid的嵌套结果，让用户检查是否合适并进行调整或者继续执行
    geo_em_files_dir = "",  # WPS输出的geo文件路径
    geogrid_picout_dir = ""
):
    shpfiles_dir = f"{WCAS_dir}input/shps/"
    shpfiles_pr = os.listdir(shpfiles_dir)
    shpfiles = []
    for f in shpfiles_pr:
        if f.split(".")[1] == "shp":
            shpfiles.append(shpfiles_dir+f)

    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(8, 6))  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    geo_file_list = os.listdir(geo_em_files_dir)  # geo文件列表
    geo_file_list.sort()
    domains_n = len(geo_file_list)  # 嵌套区域数
    domain_rectangles = []
    re_color = ['yellow']*domains_n
    re_name = ['d0x']*domains_n
    for i in geo_file_list:
        i_n = geo_file_list.index(i)
        geof = nc.Dataset(geo_em_files_dir + i)
        LAT = np.array(geof.variables['XLAT_C'][:][0])  # 读取纬度范围
        LON = np.array(geof.variables['XLONG_C'][:][0])  # 读取经度范围
        LAT_vert = getArrayVertices(LAT)  # 获取经纬度数组的四个顶点数据列表
        LON_vert = getArrayVertices(LON)
        RE = Rectangle((LON_vert[0], LAT_vert[0]), LON_vert[1] - LON_vert[0], LAT_vert[2] - LAT_vert[0], linewidth=1.3,
                       linestyle='-', zorder=2,
                       edgecolor=re_color[i_n], facecolor='none', transform=ccrs.PlateCarree())
        ax.text(LON_vert[0], LAT_vert[0], re_name[i_n], transform=ccrs.PlateCarree(), fontsize=15, c='k')
        if i_n == 0:
            d01_vert = [LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]]
        domain_rectangles.append(RE)  # 将所有矩形存放数组

    for shp_dir in shpfiles:
        shp = cfeat.ShapelyFeature(Reader(shp_dir).geometries(), proj, edgecolor='k', facecolor='none')
        ax.add_feature(shp, lw=0.5, zorder=2)
    ax.set_extent(d01_vert)  # 可根据需求自行定义
    for i in domain_rectangles:
        ax.add_patch(i)
    # --设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    plt.savefig(geogrid_picout_dir+"geogrid.png")

#ISAM相关过程
def ISAM_REGIONS_create(
    terminal = '',
    GRIDCRO2D_dir="",
    GRIDDECfile_dir="",
    Regions_varnames=[],
    target_dir="",
):
    # 将arcpy前处理程序和相关文件复制到MEIAT的ISAM目录中进行处理，因用同一python，方便整理
    terminal_exec(terminal, f'rm -rf {MEIAT_dir}ISAM') # 删除历史内容
    time.sleep(1)
    terminal_exec(terminal, f'mkdir {MEIAT_dir}ISAM')
    terminal_exec(terminal, f'cp -r {WCAS_dir}input/ISAM/maskshps {MEIAT_dir}ISAM/')
    terminal_exec(terminal, f'cp {WCAS_dir}tools/ISAMmask_ArcpyRegionsProcess_forWCAS.py {MEIAT_dir}ISAM')
    terminal_exec(terminal, f'cp {WCAS_dir}namelist.WCAS {MEIAT_dir}ISAM') # 复制过去让前处理程序读取namelist参数
    time.sleep(1)
    # 生成ISAM框架文件
    WRFCMAQ_var2tiff(
        input_ncfile_dir=GRIDCRO2D_dir,
        model_type="CMAQ",
        targetdata="DLUSE",
        isISAMtemp=True,
        outputdir=f"{MEIAT_dir}ISAM/",
    )
    # 调用arcpy的python进行处理
    terminal_exec(terminal, f"cd {MEIAT_dir}ISAM")
    terminal_exec(terminal, f"cmd.exe /c {MEIAT_python_dir} ./ISAMmask_ArcpyRegionsProcess_forWCAS.py") # 要先进入再./ 不然要必须使用Windows的盘符目录
    while (True):
        time.sleep(1)
        if os.path.exists(f"{MEIAT_dir}ISAM/ISAMmask_ArcpyRegionsProcess_forWCAS.OK") == True: # 等待运行
            break
    time.sleep(1)
    terminal_exec(terminal, f"rm -f {MEIAT_dir}ISAM/ISAMmask_ArcpyRegionsProcess_forWCAS.OK") # 删除flag文件
    #根据arcpy前处理程序生成的结果生成mask的nc文件
    ISAMMask_NCgeneration(
        GRIDDECfile_dir=GRIDDECfile_dir,
        ISAM_region_file_dir=f"{target_dir}ISAM_REGION.nc",
        Regins_tiff_dir=f"{MEIAT_dir}ISAM/Regionstifs",
        Regions_varnames = Regions_varnames,
    )
def save_2tiff(out_path, tif_data,ROW,COL,lonmin,lon_res,latmax,lat_res):
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')

    # 创建框架
    out_tif = driver.Create(out_path, COL, ROW, 1, gdal.GDT_Float32)
    # 设置影像显示范围
    geotransfor = (lonmin, lon_res, 0, latmax, 0, -lat_res)  # -latres < 0
    out_tif.SetGeoTransform(geotransfor)

    # 获取地理坐标系统信息
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # 定义输出的坐标系为"WGS 84"，AUTHORITY["EPSG","4326"]
    out_tif.SetProjection(srs.ExportToWkt())  # 赋予投影信息

    # 数据写出
    out_tif.GetRasterBand(1).WriteArray(tif_data)
    out_tif.FlushCache()
    out_tif = None
def WRFCMAQ_var2tiff(
    input_ncfile_dir = "",
    model_type = "",
    targetdata = "",
    isISAMtemp = False,
    outputdir = "x/x/",
):
    path = input_ncfile_dir
    # 读取数据
    data = nc.Dataset(path)

    # 获取经纬度
    if model_type == "WPS":
        var_lon = np.array(data.variables['CLONG'][:][0]) # WPS geog的格式
        var_lat = np.array(data.variables['CLAT'][:][0])
    elif model_type == "WRF":
        var_lon = np.array(data.variables['XLONG'][:][0][0])
        var_lat = np.array(data.variables['XLAT'][:][0][0])
    elif model_type == "CMAQ":
        var_lon = np.array(data.variables['LON'][:][0][0])
        var_lat = np.array(data.variables['LAT'][:][0][0])
    ROW = var_lon.shape[0]
    COL = var_lon.shape[1]

    # 获取影像左下角和右下角坐标
    lonmin, latmax, lonmax, latmin = (var_lon.min(), var_lat.max(),
                                      var_lon.max(), var_lat.min())
    # print(lonmin, latmax, lonmax, latmin)

    # 计算分辨率
    len_lat = ROW
    len_lon = COL
    # print(len_lon, len_lat)
    lon_res = (lonmax - lonmin) / (len_lon - 1.0)
    lat_res = (latmax - latmin) / (len_lat - 1.0)

    # 获取数据
    if model_type in ["WPS","WRF"]:
        target = np.array(data.variables[targetdata][:][0]) # WPS geog的格式
    if model_type in ["CMAQ"]:
        target = np.array(data.variables[targetdata][:][0][0])  # mcip的格式
    if isISAMtemp: # 是否输出用于制作ISAM的网格框架文件
        TEMP = np.zeros((ROW, COL), dtype=float, order="c")
        # print(TEMP.shape)
        save_2tiff(outputdir+r"temp.tif", TEMP, ROW, COL, lonmin, lon_res, latmax, lat_res)
    save_2tiff(outputdir+r"target.tif", target, ROW, COL, lonmin, lon_res, latmax, lat_res)
    # save_data(r"F:\CMAQdata\chengdu202208_d03_RegionISAM\mcip\GRIDCRO2D_2022227.tif", DLUSE,ROW,COL)
def ISAMMask_NCgeneration(
    GRIDDECfile_dir="",
    ISAM_region_file_dir="",
    Regins_tiff_dir="x/x",
    Regions_varnames = [],
):
    """
    生成供CMAQ-ISAM进行区域源解析的mask文件
    :param GRIDDECfile_dir:
    :param ISAM_region_file_dir:
    :param Regins_tiff_dir:
    :param Regions_varnames:
    :return:
    """
    # GRIDDECfile_dir = r"E:\CMAQdata_yaan202306\GRIDDESC_d02"
    # ISAM_region_file_dir = r"E:\CMAQdata_yaan202306\ISAM_REGION_d02.nc"
    # Regins_tiff_dir = r"E:\CMAQdata_yaan202306\ISAMMASK\Reginstif\\"  # 溯源区域tiff所在dir

    GRIDDECfile = open(GRIDDECfile_dir)
    GRIDDEC_info = GRIDDECfile.readlines()
    GRIDDEC_info = [line.strip() for line in GRIDDEC_info]
    range_par = GRIDDEC_info[-2].split()
    prj_par = GRIDDEC_info[2].split()
    # print(range_par,prj_par)

    # ISAM原追踪的区域
    # Regions = ['BZ','CD','DZ','DY','GA','GY','LS','LZ','MS','MY','NC','NJ','SN','YB','ZY','ZG','YA']  # ISAM原追踪的区域
    Regions = Regions_varnames
    # Regions = ['CD', 'DY', 'LS', 'MS', 'MY', 'NJ', 'SN', 'YB', 'ZY', 'ZG', 'YA']
    # Regions = ['DJY', 'CD','CZ','DY','JT','JY','LQY','PD','PJ','PZ','QBJ','QL','SL','WJ','XD','XJ']  # ISAM原追踪的区域
    Regions_str = ''
    for i in Regions:
        if len(i) == 2:
            Regions_str += i + '              '  # 14位空格
        if len(i) == 3:
            Regions_str += i + '             '  # 13位空格
        if len(i) == 4:
            Regions_str += i + '            '  # 12位空格

    # 从GRIDDEC中获取的数据 坐标系统
    NCOLS_value = int(range_par[-3])
    NROWS_value = int(range_par[-2])
    P_ALP_value = np.double(prj_par[1])
    P_BET_value = np.double(prj_par[2])
    P_GAM_value = np.double(prj_par[3])
    XCENT_value = np.double(prj_par[4])
    YCENT_value = np.double(prj_par[5])
    XORIG_value = np.double(range_par[1])
    YORIG_value = np.double(range_par[2])
    XCELL_value = np.double(range_par[3])
    YCELL_value = np.double(range_par[4])
    VGTYP = -9999
    VGTOP = 5000.0
    VGLVLS = [1.0, 0.993]
    NHOUR_value = 25
    STIME_value = 0
    TSTEP_value = 0
    NLAYS_value = 1
    NVARS_value = len(Regions)  #
    dt_num = 2
    # ts_num = 25

    # 时间变量
    CTIME_1 = time.localtime(time.time())
    CTIME_2 = [CTIME_1[0], CTIME_1[1], CTIME_1[2]]
    CTIME_3 = day2doy(CTIME_2)
    CDATE_value = str(CTIME_1[0]) + '0' + str(CTIME_3)
    CDATE_value = int(CDATE_value)
    SDATE_value = day2doy([2020, 1, 1])  # 模拟开始时间，无具体意义

    ISAM_region_file = nc.Dataset(ISAM_region_file_dir, 'w')  # 创建nc文件
    # 写入global attribute
    ISAM_region_file.IOAPI_VERSION = 'ioapi-3.2: $ld: init3.F90 185 2020-08-28 16:49:45Z coats $'
    ISAM_region_file.EXEC_ID = 'mcip'
    ISAM_region_file.FTYPE = 1
    ISAM_region_file.CDATE = CDATE_value
    ISAM_region_file.CTIME = 65043
    ISAM_region_file.WDATE = CDATE_value
    ISAM_region_file.WTIME = 65043
    ISAM_region_file.SDATE = SDATE_value
    ISAM_region_file.STIME = STIME_value
    ISAM_region_file.TSTEP = TSTEP_value
    ISAM_region_file.NTHIK = 1
    ISAM_region_file.NCOLS = NCOLS_value
    ISAM_region_file.NROWS = NROWS_value
    ISAM_region_file.NLAYS = NLAYS_value
    ISAM_region_file.NVARS = NVARS_value
    ISAM_region_file.GDTYP = 2
    ISAM_region_file.P_ALP = P_ALP_value
    ISAM_region_file.P_BET = P_BET_value
    ISAM_region_file.P_GAM = P_GAM_value
    ISAM_region_file.XCENT = XCENT_value
    ISAM_region_file.YCENT = YCENT_value
    ISAM_region_file.XORIG = XORIG_value
    ISAM_region_file.YORIG = YORIG_value
    ISAM_region_file.XCELL = XCELL_value
    ISAM_region_file.YCELL = YCELL_value
    ISAM_region_file.VGTYP = VGTYP
    ISAM_region_file.VGTOP = VGTOP
    ISAM_region_file.VGLVLS = VGLVLS
    ISAM_region_file.GDNAM = 'CMAQ ISAM mask'
    ISAM_region_file.UPNAM = 'made in gaoc'
    ISAM_region_file.FILEDESC = 'made in gaoc'
    ISAM_region_file.HISTORY = 'made in gaoc'
    ISAM_region_file.VAR_LIST = Regions_str
    ISAM_region_file.renameAttribute("VAR_LIST", "VAR-LIST")  # python中不能直接.VAR-LIST ‘-’会报错

    # 写入Dimension变量
    ts_id = ISAM_region_file.createDimension('TSTEP', 1)  #
    dt_id = ISAM_region_file.createDimension('DATE-TIME', dt_num)  # len(lat)
    l_id = ISAM_region_file.createDimension('LAY', NLAYS_value)
    v_id = ISAM_region_file.createDimension('VAR', NVARS_value)
    r_id = ISAM_region_file.createDimension('ROW', NROWS_value)
    c_id = ISAM_region_file.createDimension('COL', NCOLS_value)

    # 写入Variables
    TFLAG_id = ISAM_region_file.createVariable('TFLAG', 'l', (ts_id, v_id, dt_id))  # 必须指明变量类型
    TFLAG_id.units = '<YYYYDDD,HHMMSS>'  # 给variable写attribute的方法
    TFLAG_id.long_name = 'TFLAG'
    TFLAG_id.var_desc = 'Timestep-valid flags: (1) YYYYDDD or (2) HHMMSS'
    TFLAG_id[:] = 0  # 必须把这个设置为0，来说明所有变量都是time-independent，否则CMAQ会报错
    # print(STIME_value)

    # 获取溯源区域tif
    Regions_tif_file_list_pr = os.listdir(Regins_tiff_dir)  # 获得WRF文件列表
    Regions_tif_file_list = []
    # WRF_file_list_time = {}
    for i in Regions_tif_file_list_pr:
        if i[-4:] == '.tif':
            Regions_tif_file_list.append(i)

    for i in Regions_tif_file_list:
        target_data = gdal.Open(Regins_tiff_dir + '/' + i)
        REGION_name = os.path.basename(Regins_tiff_dir + '/' + i).split(".tif")[0]
        # print(REGION_name)
        data_box_col = target_data.RasterYSize  # 获取列行
        data_box_line = target_data.RasterXSize
        geoTransform = target_data.GetGeoTransform()  # 获取geoinfo
        data = target_data.ReadAsArray(0, 0, data_box_line, data_box_col)  # 读取数据存放到创建的数组中
        data = np.flipud(data)  # 经CMAQ输出发现，MASK图像要翻转一下
        data[data != 0] = 1  # mask为1 其他为0
        data[data == 0] = 0
        # data[:,78] = 0 #误差处理
        REGION = ISAM_region_file.createVariable(REGION_name, 'f', (ts_id, l_id, r_id, c_id))  # 必须指明变量类型
        REGION[:] = data  # 给创建的netcdfvariable给值，即是区域tiff的01值
        REGION.units = 'fraction'  # 给variable写attribute的方法
        REGION.TSETP = 1
        REGION.long_name = REGION_name
        REGION.var_desc = REGION_name + ' fractional area per grid cell'
    ISAM_region_file.close()   # 在WCAS流程制作ISAMmask不知为何必须关闭，不然不能完整输出

#后处理过程
def O3PM25Combine_CMAQfiles(
    CCTM_dir = "", # 需要合并的已经分类的CCTM文件夹路径
    Combine_file_dir = "",  # 合并后的文件名称，必须自己创建combine文件夹
    GRIDDECfile_dir = "",
    start_time_yesterday = [2024, 3, 14], # 开始日期的前一天
):
    
    Combine_substance = ['ASO4I','ANO3I','ANH4I','ANAI','ACLI','AECI','AOTHRI',# 合并的PM2.5组分
                        'APOCI','APNCOMI','ALVOO1I','ALVOO2I','ASVOO1I','ASVOO2I',
                        'ALVPO1I','ASVPO1I','ASVPO2I',

                        'ASO4J','ANO3J','ANH4J','ANAJ','ACLJ','AXYL1J','AXYL2J',
                        'AXYL3J','ATOL1J','ATOL2J','ATOL3J','ABNZ1J','ABNZ2J',
                        'ABNZ3J','AISO1J','AISO2J','AISO3J','ATRP1J','ATRP2J',
                        'ASQTJ','AALK1J','AALK2J','APAH1J','APAH2J','APAH3J',
                        'AORGCJ','AOLGBJ','AOLGAJ','ALVOO1J','ALVOO2J','ASVOO1J',
                        'ASVOO2J','ASVOO3J','APCSOJ','ALVPO1J','ASVPO1J','ASVPO2J',
                        'ASVPO3J','AIVPO1J','AOTHRJ','AFEJ','ASIJ','ATIJ','ACAJ',
                        'AMGJ','AMNJ','AALJ','AKJ',

                        'ASOIL','ACORS','ASEACAT','ACLK','ASO4K','ANO3K','ANH4K',

                        'O3'                                                       # 臭氧
                        ]
    PM25_para = ['PM25AT','PM25AC','PM25CO'] # 计算PM2.5浓度的参数，在CCTM_APMDIAG文件中

    combine_file_namestr = f'ACONCv{CCTMversion}'  # 合并的CCTM文件名称类型，CCTM_ 之后的2个_隔开的名称相加
    # combine_file_namestr = 'PA1'  # 合并的CCTM文件名称类型，CCTM_ 之后的2个_隔开的名称相加
    daynum = len(os.listdir(CCTM_dir))/2 # 合并的文件个数，由于包含APMDIAG需要/2

    SA_CCTM_file_list = os.listdir(CCTM_dir)  # CCTM
    to_combine_files_list = []
    APMDIAG_files_list = []
    for i in SA_CCTM_file_list:  # 找出对应文件列表
        if i[-3:] == '.nc':
            if i.split("_")[1] + i.split("_")[2] == combine_file_namestr:
                to_combine_files_list.append(i)
            if i.split("_")[1] == 'APMDIAG':
                APMDIAG_files_list.append(i)
    # print(to_combine_files_list)

    # 合并文件
    CONCf_pre = nc.Dataset(CCTM_dir + to_combine_files_list[0])  # 先打开一个文件，获取数据信息
    all_vars_pre = list(CONCf_pre.variables.keys())
    # print(all_vars_pre)
    all_vars = []
    # print(len(Combine_substance)-1)
    for x in all_vars_pre:
        for m in Combine_substance:
            if x[0:len(m)] in Combine_substance and x not in all_vars:  # 查找物质，包含且不重复
            # if x[-len(m):] in Combine_substance and x not in all_vars:  # 查找物质，包含且不重复，PA文件命名特殊，要从后面获取
            #     print(x)
                all_vars.append(x)
    for x in PM25_para: # 把计算参数加上
            all_vars.append(x)
    # all_vars.remove('O3_ AANDB') # 只得到A

    # print(all_vars)
    all_vars_lists = {}

    for i in all_vars:  # 给所有变量创建空列表，以放入合并后的总arr {'ANO3J_DJYEMI': [], 'ANO3I_DJYEMI': [], 'HNO3_DJYEMI': [], ......}
        all_vars_lists.update({i: []})
    # print(all_vars_lists)

    CCTM_day = len(to_combine_files_list)
    to_combine_files_list.sort()  # 按照时间排序
    APMDIAG_files_list.sort()
    print(T.Now() + ' - ' +"PM25/O3组分数据读取合并...")
    time.sleep(0.5)  # 多次进行进度条调用时防止过快导致的显示异常
    for x in tqdm(all_vars,desc=T.Now() + ' - '):
        if x in Combine_substance: # 合并组分
            for i in to_combine_files_list:
                CONCf = nc.Dataset(CCTM_dir + i)
                day = to_combine_files_list.index(i) + 1  # 合并到第几天
                n_next = to_combine_files_list.index(i) + 1  # 下一天文件的index
                if n_next >= CCTM_day:
                    all_vars_lists.update({x: data_now}) # 写入合并后的数据
                    break
                CONCf_next = nc.Dataset(CCTM_dir + to_combine_files_list[n_next])

                data_next = np.array(CONCf_next.variables[x][:])
                if to_combine_files_list.index(i) == 0:
                    data_now = np.array(CONCf.variables[x][:]) # 首次输入文件
                    data_now = np.concatenate((data_now, data_next), axis=0) # ACONC没有多的1h，不用删除处理

                if to_combine_files_list.index(i) != 0:
                    data_now = np.concatenate((data_now, data_next), axis=0)
        if x in PM25_para: # 合并PM2.5计算参数
            for i in APMDIAG_files_list:
                CONCf = nc.Dataset(CCTM_dir + i)
                day = APMDIAG_files_list.index(i) + 1  # 合并到第几天
                n_next = APMDIAG_files_list.index(i) + 1  # 下一天文件的index
                if n_next >= CCTM_day:
                    all_vars_lists.update({x: data_now})  # 写入合并后的数据
                    break
                CONCf_next = nc.Dataset(CCTM_dir + APMDIAG_files_list[n_next])

                data_next = np.array(CONCf_next.variables[x][:])
                if APMDIAG_files_list.index(i) == 0:
                    data_now = np.array(CONCf.variables[x][:])  # 首次输入文件
                    data_now = np.concatenate((data_now, data_next), axis=0)  # ACONC没有多的1h，不用删除处理

                if APMDIAG_files_list.index(i) != 0:
                    data_now = np.concatenate((data_now, data_next), axis=0)
    #     print(x,' 物种合并完成')
    # print(x, ' ACONC合并完成')


    GRIDDECfile = open(GRIDDECfile_dir)
    GRIDDEC_info = GRIDDECfile.readlines()
    GRIDDEC_info = [line.strip() for line in GRIDDEC_info]
    range_par = GRIDDEC_info[-2].split()
    prj_par = GRIDDEC_info[2].split()
    # print(range_par,prj_par)

    dt_num = daynum
    ts_num = daynum * 24
    VAR_LIST_l = ''
    for i in all_vars:  # 制作每个变量16位的varlist
        d_len = 16 - len(i)
        i += d_len * ' '
        VAR_LIST_l += i
    # print(VAR_LIST_l)

    # 从GRIDDEC中获取的数据 坐标系统
    NCOLS_value = int(range_par[-3])
    NROWS_value = int(range_par[-2])
    P_ALP_value = np.double(prj_par[1])
    P_BET_value = np.double(prj_par[2])
    P_GAM_value = np.double(prj_par[3])
    XCENT_value = np.double(prj_par[4])
    YCENT_value = np.double(prj_par[5])
    XORIG_value = np.double(range_par[1])
    YORIG_value = np.double(range_par[2])
    XCELL_value = np.double(range_par[3])
    YCELL_value = np.double(range_par[4])
    VGTYP = -9999
    VGTOP = 5000.0
    VGLVLS = [1.0, 0.993]
    # NHOUR_value = 25
    STIME_value = 0  # 起始时刻的分钟秒钟
    TSTEP_value = 10000  # 1h输出间隔
    NLAYS_value = 1
    NVARS_value = len(all_vars)  #
    dt_num = 2

    # 时间变量
    CTIME_1 = time.localtime(time.time())
    CTIME_2 = [CTIME_1[0], CTIME_1[1], CTIME_1[2]]
    CTIME_3 = day2doy(CTIME_2)
    CDATE_value = str(CTIME_1[0]) + '0' + str(CTIME_3)
    CDATE_value = int(CDATE_value)
    SDATE_value = day2doy(start_time_yesterday)  # 模拟开始时间的前一天，不知为何要前一天

    COMBINE_file = nc.Dataset(Combine_file_dir, 'w')  # 创建nc文件
    # 写入global attribute
    COMBINE_file.IOAPI_VERSION = 'ioapi-3.2: $ld: init3.F90 185 2020-08-28 16:49:45Z coats $'
    COMBINE_file.EXEC_ID = 'mcip'
    COMBINE_file.FTYPE = 1
    COMBINE_file.CDATE = CDATE_value
    COMBINE_file.CTIME = 65043
    COMBINE_file.WDATE = CDATE_value
    COMBINE_file.WTIME = 65043
    COMBINE_file.SDATE = SDATE_value
    COMBINE_file.STIME = STIME_value
    COMBINE_file.TSTEP = TSTEP_value
    COMBINE_file.NTHIK = 1
    COMBINE_file.NCOLS = NCOLS_value
    COMBINE_file.NROWS = NROWS_value
    COMBINE_file.NLAYS = NLAYS_value
    COMBINE_file.NVARS = NVARS_value
    COMBINE_file.GDTYP = 2
    COMBINE_file.P_ALP = P_ALP_value
    COMBINE_file.P_BET = P_BET_value
    COMBINE_file.P_GAM = P_GAM_value
    COMBINE_file.XCENT = XCENT_value
    COMBINE_file.YCENT = YCENT_value
    COMBINE_file.XORIG = XORIG_value
    COMBINE_file.YORIG = YORIG_value
    COMBINE_file.XCELL = XCELL_value
    COMBINE_file.YCELL = YCELL_value
    COMBINE_file.VGTYP = VGTYP
    COMBINE_file.VGTOP = VGTOP
    COMBINE_file.VGLVLS = VGLVLS
    COMBINE_file.GDNAM = 'CCTM_SA_ACONC_COMBINE'
    COMBINE_file.UPNAM = 'made in gaoc'
    COMBINE_file.FILEDESC = 'made in gaoc'
    COMBINE_file.HISTORY = 'made in gaoc'
    COMBINE_file.VAR_LIST = VAR_LIST_l
    COMBINE_file.renameAttribute("VAR_LIST", "VAR-LIST")  # python中不能直接.VAR-LIST ‘-’会报错

    # 写入Dimension变量
    ts_id = COMBINE_file.createDimension('TSTEP', ts_num)  #
    dt_id = COMBINE_file.createDimension('DATE-TIME', dt_num)  # len(lat)
    l_id = COMBINE_file.createDimension('LAY', NLAYS_value)
    v_id = COMBINE_file.createDimension('VAR', NVARS_value)
    r_id = COMBINE_file.createDimension('ROW', NROWS_value)
    c_id = COMBINE_file.createDimension('COL', NCOLS_value)

    # 写入TFLAG
    TFLAG_id = COMBINE_file.createVariable('TFLAG', 'l', (ts_id, v_id, dt_id))  # 必须指明变量类型
    TFLAG_id.units = '<YYYYDDD,HHMMSS>'  # 给variable写attribute的方法
    TFLAG_id.long_name = 'TFLAG'
    TFLAG_id.var_desc = 'Timestep-valid flags: (1) YYYYDDD or (2) HHMMSS'
    # TFLAG_id[0,:,:] = SDATE_value
    # print(STIME_value)
    # for i in all_vars_lists:
    #     # print('awdawdad                ',i)

    print(T.Now() + ' - ' +"导入combine后的nc文件...")
    time.sleep(0.5)  # 多次进行进度条调用时防止过快导致的显示异常
    for i in tqdm(all_vars_lists,desc=T.Now() + ' - '):
        data_id = COMBINE_file.createVariable(i, 'f', (ts_id, l_id, r_id, c_id))  # 必须指明变量类型
        data_id.long_name = i + (16 - len(i)) * ' '
        if i in ['PM25AT', 'PM25AC']:
            data_id.units = ' '  # 计算系数无单位
        else:
            data_id.units = 'ug/m-3'
        data_id.var_desc = i
        data_id[:] = all_vars_lists[i]
        # print('物种 ',i,' nc文件导入完成')

    # 计算PM25浓度并加入
    data_shape = np.array(COMBINE_file.variables[all_vars[0]][:]).shape  # 获得数据shape来存放all
    PM25_data = np.zeros(data_shape, 'float64')  # PM25数组初始化
    print(T.Now() + ' - ' +f"PM25每小时浓度计算...")
    time.sleep(0.5)  # 多次进行进度条调用时防止过快导致的显示异常
    for t in tqdm(range(0, data_shape[0]),desc=T.Now() + ' - '):  # 每个小时的数据相加
        PM25_AT_data = np.array(COMBINE_file.variables['PM25AT'][:])
        PM25_AC_data = np.array(COMBINE_file.variables['PM25AC'][:])
        PM25_CO_data = np.array(COMBINE_file.variables['PM25CO'][:])
        PM25_AI_data = np.zeros(data_shape, 'float64')
        PM25_AJ_data = np.zeros(data_shape, 'float64')  # PM25数组初始化
        PM25_AK_data = np.zeros(data_shape, 'float64')
        for i in all_vars_lists:
            if i not in ['PM25AT', 'PM25AC', 'PM25CO']:
                if i in ['ASO4I', 'ANO3I', 'ANH4I', 'ANAI', 'ACLI', 'AECI', 'AOTHRI',
                        'APOCI', 'APNCOMI', 'ALVOO1I', 'ALVOO2I', 'ASVOO1I', 'ASVOO2I',
                        'ALVPO1I', 'ASVPO1I', 'ASVPO2I']:
                    PM_sub_data = np.array(COMBINE_file.variables[i][:])
                    # PM25_AI_data[t, :, :, :] = parallel_compute_add(PM25_AI_data[t, :, :, :], PM_sub_data[t, :, :, :], cores_win)
                    PM25_AI_data[t, :, :, :] += PM_sub_data[t, :, :, :]
                if i in ['ASO4J', 'ANO3J', 'ANH4J', 'ANAJ', 'ACLJ', 'AXYL1J', 'AXYL2J',
                        'AXYL3J', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J',
                        'ABNZ3J', 'AISO1J', 'AISO2J', 'AISO3J', 'ATRP1J', 'ATRP2J',
                        'ASQTJ', 'AALK1J', 'AALK2J', 'APAH1J', 'APAH2J', 'APAH3J',
                        'AORGCJ', 'AOLGBJ', 'AOLGAJ', 'ALVOO1J', 'ALVOO2J', 'ASVOO1J',
                        'ASVOO2J', 'ASVOO3J', 'APCSOJ', 'ALVPO1J', 'ASVPO1J', 'ASVPO2J',
                        'ASVPO3J', 'AIVPO1J', 'AOTHRJ', 'AFEJ', 'ASIJ', 'ATIJ', 'ACAJ',
                        'AMGJ', 'AMNJ', 'AALJ', 'AKJ']:
                    PM_sub_data = np.array(COMBINE_file.variables[i][:])
                    # PM25_AJ_data[t, :, :, :] = parallel_compute_add(PM25_AJ_data[t, :, :, :], PM_sub_data[t, :, :, :], cores_win)
                    PM25_AJ_data[t, :, :, :] += PM_sub_data[t, :, :, :]
                if i in ['ASOIL', 'ACORS', 'ASEACAT', 'ACLK', 'ASO4K', 'ANO3K', 'ANH4K']:
                    PM_sub_data = np.array(COMBINE_file.variables[i][:])
                    # PM25_AK_data[t, :, :, :] = parallel_compute_add(PM25_AK_data[t, :, :, :], PM_sub_data[t, :, :, :], cores_win)
                    PM25_AK_data[t, :, :, :] += PM_sub_data[t, :, :, :]
        # AIAT = parallel_compute_multipy(PM25_AI_data[t, :, :, :],PM25_AT_data[t, :, :, :],cores_win)
        # AJAC = parallel_compute_multipy(PM25_AJ_data[t, :, :, :],PM25_AC_data[t, :, :, :],cores_win)
        # AKCO = parallel_compute_multipy(PM25_AK_data[t, :, :, :],PM25_CO_data[t, :, :, :],cores_win)
        # PM25_data[t, :, :, :] = parallel_compute_add(parallel_compute_add(AIAT,AJAC,cores_win),AKCO,cores_win)
        PM25_data[t, :, :, :] = PM25_AI_data[t, :, :, :] * PM25_AT_data[t, :, :, :] + \
                                PM25_AJ_data[t, :, :, :] * PM25_AC_data[t, :, :, :] + \
                                PM25_AK_data[t, :, :, :] * PM25_CO_data[t, :, :, :]

    data_id = COMBINE_file.createVariable('PM25', 'f', (ts_id, l_id, r_id, c_id))  # 必须指明变量类型
    data_id.long_name = 'PM25' + (16 - len('PM25')) * ' '
    data_id.units = 'ug/m-3'
    data_id.var_desc = 'PM25'
    data_id[:] = PM25_data
def WRFdata_output(
    WRFoutd03_files_dir="",
    WRFoutd02_files_dir="",
    WCAS_output_dir = "",
    start_date = "YYYY-MM-DD-HH"
):
    shpfiles_dir = f"{WCAS_dir}input/shps/"
    shpfiles_pr = os.listdir(shpfiles_dir)
    shpfiles = []
    for f in shpfiles_pr:
        if f.split(".")[1] == "shp":
            shpfiles.append(shpfiles_dir+f)
    deg = 180.0 / np.pi
    rad = np.pi / 180.0

    WRF_file_list_D03 = os.listdir(WRFoutd03_files_dir)  # 获得WRF文件列表
    WRF_file_list_D02 = os.listdir(WRFoutd02_files_dir)  # 获得WRF文件列表
    WRF_file_list_D03.sort()
    WRF_file_list_D02.sort()
    WRF_file_list_D02.pop() # WRFout的最后一个文件不在目标模拟天数内，去除
    WRF_file_list_D03.pop() 
    result_hours = len(WRF_file_list_D03) * 24  # 从开始时期输出多少小时数据结果，根据文件数量
    result_days = len(WRF_file_list_D03)


    WRFoutf = h5py.File(WRFoutd03_files_dir + WRF_file_list_D03[0], "r")  # 打开WRF输出的HDF格式文件
    ROWd03 = np.array(WRFoutf.get("XLAT")).shape[1]  # 获得数据格式shape1
    COLd03 = np.array(WRFoutf.get("XLAT")).shape[2]  # 获得数据格式shape2
    WRFoutf.close()
    WRFoutf = h5py.File(WRFoutd02_files_dir + WRF_file_list_D02[0], "r")  # 打开WRF输出的HDF格式文件
    ROWd02 = np.array(WRFoutf.get("XLAT")).shape[1]  # 获得数据格式shape1
    COLd02 = np.array(WRFoutf.get("XLAT")).shape[2]  # 获得数据格式shape2
    WRFoutf.close()

    print(T.Now() + ' - ' +f"处理逐小时气象场...")
    T2_daymean = np.zeros((1, ROWd03, COLd03), dtype=np.float64) # 同时输出日均值以及模拟时段的总均值
    T2_allmean = np.zeros((1, ROWd03, COLd03), dtype=np.float64)
    PBLH_daymean = np.zeros((1, ROWd03, COLd03), dtype=np.float64) 
    PBLH_allmean = np.zeros((1, ROWd03, COLd03), dtype=np.float64)
    RH_daymean = np.zeros((1, ROWd03, COLd03), dtype=np.float64) 
    RH_allmean = np.zeros((1, ROWd03, COLd03), dtype=np.float64)
    WS_V_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
    WS_V_allmean = np.zeros((1, ROWd02, COLd02), dtype=np.float64)
    WS_U_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
    WS_U_allmean = np.zeros((1, ROWd02, COLd02), dtype=np.float64)
    WD_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
    WD_allmean = np.zeros((1, ROWd02, COLd02), dtype=np.float64)
    for UTC_hour in tqdm(range(0, result_hours),desc=T.Now() + ' - '):
        true_hour = UTC_hour+8
        date_now = str(datetime.datetime.strptime(start_date, '%Y-%m-%d-%H')+datetime.timedelta(hours=true_hour))
        date_now_UTC = str(datetime.datetime.strptime(start_date, '%Y-%m-%d-%H')+datetime.timedelta(hours=UTC_hour))
        if UTC_hour == 0:
            UTC_day = int(date_now_UTC.split('-')[2].split(' ')[0])-int(start_date.split("-")[2]) # 用于判断该处理第几个WRFout
        if UTC_hour in range(0,result_hours,24) and UTC_hour != 0:
            UTC_day += 1

        WRFoutf = h5py.File(WRFoutd03_files_dir + WRF_file_list_D03[UTC_day], "r")  # 打开第几天WRF输出的HDF格式文件
        WRFoutfD02 = h5py.File(WRFoutd02_files_dir + WRF_file_list_D02[UTC_day], "r")  # 打开第几天WRF输出的HDF格式文件
        XLAT = np.array(WRFoutf.get("XLAT"))  # 第三层经纬度
        XLONG = np.array(WRFoutf.get("XLONG"))
        XLATD02 = np.array(WRFoutfD02.get("XLAT"))  # 第二层经纬度 风场用
        XLONGD02 = np.array(WRFoutfD02.get("XLONG"))
        LAT_vert = getArrayVertices(XLAT[0])  # 获取经纬度数组的四个顶点数据列表
        LON_vert = getArrayVertices(XLONG[0])
        if MannualExtent == True:
            extent = Extent
        else:
            extent = [LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]]

        T2_K = np.array(WRFoutf.get("T2"))  # 从WRFout得到地面2m温度 单位为K
        T2 = T2_K - 273.15  # K转化为摄氏度

        PBLH = np.array(WRFoutf.get("PBLH"))  # 边界层高度

        P = np.array(WRFoutf.get("PSFC"))  # 气压 单位为Pa
        SH = np.array(WRFoutf.get("Q2"))  # 比湿
        RH = 0.236 * P * SH * np.exp((17.67 * T2) / (T2_K - 29.65)) ** (-1)  # 相对湿度

        WS_V = np.array(WRFoutfD02.get("V10"))  # 风速经向分量
        WS_U = np.array(WRFoutfD02.get("U10"))  # 风速纬向分量
        WD = 180.0 + np.arctan2(WS_U, WS_V) * deg  # 计算风向
        UTC_hour_var = UTC_hour-24*UTC_day # 每个文件只有24h
        winddata = [XLONGD02[UTC_hour_var, :, :], XLATD02[UTC_hour_var, :, :],
                     WS_V[UTC_hour_var, :, :], WS_U[UTC_hour_var, :, :] ,WD[UTC_hour_var, :, :]]
        
        T2_daymean += T2[UTC_hour_var, :, :]
        PBLH_daymean += PBLH[UTC_hour_var, :, :]
        RH_daymean += RH[UTC_hour_var, :, :]
        WS_V_daymean += WS_V[UTC_hour_var, :, :]
        WS_U_daymean += WS_U[UTC_hour_var, :, :]
        WD_daymean += WD[UTC_hour_var, :, :]

        WRF_outdata,WRF_outdata_str, = [T2,PBLH,RH],['T2','PBLH','RH']
        WRF_outdata_daymean = [T2_daymean,PBLH_daymean,RH_daymean]
        WRF_outdata_allmean = [T2_allmean,PBLH_allmean,RH_allmean]
        barlabels = ['temperature (℃)',"PBLH Height(m)","RH"]
        cmaps = ['rainbow','jet','ocean']
        index = 0
        for outdata in WRF_outdata: # 逐小时输出各个变量
            outdata_str = WRF_outdata_str[index]
            title = f"Hourly {WRF_outdata_str[index]} at {date_now}"
            cbarmax = np.ceil(np.max(outdata[:]) / 10) * 10
            cbarmin = np.floor(np.min(outdata[:]) / 10) * 10
            if abs(np.max(outdata[:])) <= 10:
                cbarmax = np.max(outdata[:])
            if abs(np.min(outdata[:])) <= 10:
                cbarmin = np.min(outdata[:])
            if cbarmin == 0 and cbarmax == 0: # 部分情况无值
                cbarmax,cbarmin = 1,-1
            barinterval = np.floor((cbarmax - cbarmin)/5)
            os.makedirs(WCAS_output_dir+f"{outdata_str}/Hourly/",exist_ok=True)
            pic_output = WCAS_output_dir+f"{outdata_str}/Hourly/"
            drawCoutourf(
                outdata[UTC_hour_var, :, :], XLONG[UTC_hour_var, :, :], 
                XLAT[UTC_hour_var, :, :], shpfiles, 
                title, pic_output, cmaps[index], extent,barinterval,
                'WRF_hourly',winddatas=winddata,barmin=cbarmin,barmax=cbarmax,
                barlabel=barlabels[index]
            )
            index +=1
        if UTC_hour in range(16,result_hours-24+16,24): # 每过一天输出一次日均，无模拟时段(比目标时段长)的第一天和最后一天的均值
            index = 0
            WS_V_allmean += WS_V_daymean
            WS_U_allmean += WS_U_daymean
            WD_allmean += WD_daymean
            WS_V_daymean/=24
            WS_U_daymean/=24
            WD_daymean/=24
            for outdata in WRF_outdata:
                outdata_str = WRF_outdata_str[index]
                outdata_daymean = WRF_outdata_daymean[index]
                WRF_outdata_allmean[index]  += outdata_daymean
                os.makedirs(WCAS_output_dir+f"{outdata_str}/daymean/",exist_ok=True)
                pic_output_daymean = WCAS_output_dir+f"{outdata_str}/daymean/"
                outdata_daymean[:] /= 24
                cbarmax = getCbarmax(outdata_daymean[:])[0]
                cbarmin = getCbarmax(outdata_daymean[:])[1]
                barinterval = (cbarmax - cbarmin)/5
                date_yesterday = str(datetime.datetime.strptime(date_now.split(' ')[0], '%Y-%m-%d')) # 每次是当前昨天的均值
                # date_yesterday = str(datetime.datetime.strptime(date_now.split(' ')[0], '%Y-%m-%d')-datetime.timedelta(days=1)) # 每次是当前昨天的均值
                title = f"Daymean {WRF_outdata_str[index]} at {date_yesterday}"
                winddata_daymean = [XLONGD02[:], XLATD02[:],WS_V_daymean[:], WS_U_daymean[:] ,WD_daymean[:]]
                drawCoutourf(
                    outdata_daymean[:], XLONG[:][0], 
                    XLAT[:][0], shpfiles, 
                    title, pic_output_daymean, cmaps[index], extent,barinterval,
                    'WRF',winddatas=winddata_daymean,barmin=cbarmin,barmax=cbarmax,
                    barlabel=barlabels[index]
                )
                WRF_outdata_daymean[index] = np.zeros((1, ROWd03, COLd03), dtype=np.float64) # 重置
                index +=1
            WS_V_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
            WS_U_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
            WD_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
        if UTC_hour == result_hours-24+16: # 最后一天，输出完整的模拟天的平均值
            index = 0
            WS_V_allmean/=result_hours-24
            WS_U_allmean/=result_hours-24
            WD_allmean/=result_hours-24
            for outdata in WRF_outdata:
                outdata_allmean = WRF_outdata_allmean[index]
                os.makedirs(WCAS_output_dir+f"{outdata_str}/",exist_ok=True)
                pic_output_allmean = WCAS_output_dir
                outdata_allmean[:] /= result_hours-24
                title = f"WRF simulation mean {WRF_outdata_str[index]}"
                winddata_allmean = [XLONGD02[:], XLATD02[:],WS_V_allmean[:], WS_U_allmean[:] ,WD_allmean[:]]
                cbarmax = getCbarmax(outdata_allmean[:])[0]
                cbarmin = getCbarmax(outdata_allmean[:])[1]
                barinterval = (cbarmax - cbarmin)/5
                drawCoutourf(
                    outdata_allmean[:], XLONG[:][0], 
                    XLAT[:][0], shpfiles, 
                    title, pic_output_allmean, cmaps[index], extent,barinterval,
                    'WRF',winddatas=winddata_allmean,barmin=cbarmin,barmax=cbarmax,
                    barlabel=barlabels[index]
                )
                index +=1
def CMAQdata_output(
    CombineFile_dir="",
    GRIDCRO2D_dir="",
    WRFoutd02_files_dir="",
    WCAS_output_dir = "",
    daynum = 1,
    start_date = "YYYY-MM-DD-HH"
):
    shpfiles_dir = f"{WCAS_dir}input/shps/"
    shpfiles_pr = os.listdir(shpfiles_dir)
    shpfiles = []
    for f in shpfiles_pr:
        if f.split(".")[1] == "shp":
            shpfiles.append(shpfiles_dir+f)
    deg = 180.0 / np.pi
    rad = np.pi / 180.0

    WRF_file_list_D02 = os.listdir(WRFoutd02_files_dir)  # 用于绘制风场
    WRF_file_list_D02.sort()
    WRF_file_list_D02.pop() # WRFout的最后一个文件不在目标模拟天数内，去除
    WRF_file_list_D02.remove(WRF_file_list_D02[0]) # CMAQ起始时间比WRF晚一天
    WRFoutf = h5py.File(WRFoutd02_files_dir + WRF_file_list_D02[0], "r")  # 打开WRF输出的HDF格式文件
    ROWd02 = np.array(WRFoutf.get("XLAT")).shape[1]  # 获得数据格式shape1
    COLd02 = np.array(WRFoutf.get("XLAT")).shape[2]  # 获得数据格式shape2
    WRFoutf.close()

    # 分辨率计算,经纬度范围计算
    geof = nc.Dataset(GRIDCRO2D_dir)
    Combinefile = nc.Dataset(CombineFile_dir)
    LAT = np.array(geof.variables['LAT'][:][0][0])  # 读取纬度范围
    LON = np.array(geof.variables['LON'][:][0][0])  # 读取经度范围
    LAT_vert = getArrayVertices(LAT)  # 获取经纬度数组的四个顶点数据列表
    LON_vert = getArrayVertices(LON)
    if MannualExtent == True:
        extent = Extent
    else:
        extent = [LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]]
    ROW = LON.shape[0]
    COL = LON.shape[1]
    LAT_vert[0] += 0.01  # 消除图像下侧因投影产生的空隙区

    O3_daymean = np.zeros((1, ROW, COL), dtype=np.float64) 
    O3_allmean = np.zeros((1, ROW, COL), dtype=np.float64)
    PM25_daymean = np.zeros((1, ROW, COL), dtype=np.float64) 
    PM25_allmean = np.zeros((1, ROW, COL), dtype=np.float64)
    WS_V_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
    WS_V_allmean = np.zeros((1, ROWd02, COLd02), dtype=np.float64)
    WS_U_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
    WS_U_allmean = np.zeros((1, ROWd02, COLd02), dtype=np.float64)
    WD_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
    WD_allmean = np.zeros((1, ROWd02, COLd02), dtype=np.float64)
    
    result_hours = int(daynum*24)
    for UTC_hour in tqdm(range(0,result_hours),desc=T.Now() + ' - '):
        true_hour = UTC_hour+8
        date_now = str(datetime.datetime.strptime(start_date, '%Y-%m-%d-%H')+datetime.timedelta(hours=true_hour))
        date_now_UTC = str(datetime.datetime.strptime(start_date, '%Y-%m-%d-%H')+datetime.timedelta(hours=UTC_hour))
        if UTC_hour == 0:
            UTC_day = int(date_now_UTC.split('-')[2].split(' ')[0])-int(start_date.split("-")[2]) # 用于判断该处理第几个WRFout
        if UTC_hour in range(0,result_hours,24) and UTC_hour != 0:
            UTC_day += 1

        WRFoutfD02 = h5py.File(WRFoutd02_files_dir + WRF_file_list_D02[UTC_day], "r")  # 打开第几天WRF输出的HDF格式文件
        XLATD02 = np.array(WRFoutfD02.get("XLAT"))  # 第二层经纬度 风场用
        XLONGD02 = np.array(WRFoutfD02.get("XLONG"))
        WS_V = np.array(WRFoutfD02.get("V10"))  # 风速经向分量
        WS_U = np.array(WRFoutfD02.get("U10"))  # 风速纬向分量
        WD = 180.0 + np.arctan2(WS_U, WS_V) * deg  # 计算风向
        UTC_hour_var = UTC_hour-24*UTC_day # 每个文件只有24h
        winddata = [XLONGD02[UTC_hour_var, :, :], XLATD02[UTC_hour_var, :, :],
                     WS_V[UTC_hour_var, :, :], WS_U[UTC_hour_var, :, :] ,WD[UTC_hour_var, :, :]]
        
        O3 = np.array(Combinefile.variables["O3"][:])[UTC_hour, 0, :, :] * (48 / 22.4) * 1000 # CMAQ变量经过了combine，可用总UTC_hour
        PM25 = np.array(Combinefile.variables["PM25"][:])[UTC_hour, 0, :, :] 

        O3_daymean += O3
        PM25_daymean += PM25
        WS_V_daymean += WS_V[UTC_hour_var, :, :]
        WS_U_daymean += WS_U[UTC_hour_var, :, :]
        WD_daymean += WD[UTC_hour_var, :, :]

        CMAQ_outdata,CMAQ_outdata_str, = [O3,PM25],['O$_{3}$','PM$_{2.5}$',]
        CMAQ_outdata_daymean = [O3_daymean,PM25_daymean]
        CMAQ_outdata_allmean = [O3_allmean,PM25_allmean]
        barlabels = ['ug/m$^{3}$',"ug/m$^{3}$"]
        cmapdict = ['white', '#75bbfd', 'green', 'yellow', 'red', 'maroon']  # 自定义colorbar的颜色
        mycmap = colors.LinearSegmentedColormap.from_list("name", cmapdict)
        cmaps = [mycmap,mycmap]
        index = 0
        for outdata in CMAQ_outdata: # 逐小时输出各个变量
            outdata_str = CMAQ_outdata_str[index]
            title = f"Hourly {CMAQ_outdata_str[index]} at {date_now}"
            cbarmax = getCbarmax(outdata[:])[0]
            cbarmin = getCbarmax(outdata[:])[1]
            barinterval = np.ceil((cbarmax - cbarmin)/5)
            os.makedirs(WCAS_output_dir+f"{outdata_str}/Hourly/",exist_ok=True)
            pic_output = WCAS_output_dir+f"{outdata_str}/Hourly/"
            drawCoutourf(
                outdata, LON[:, :], 
                LAT[:, :], shpfiles, 
                title, pic_output, cmaps[index], extent,barinterval,
                'CMAQ_hourly',winddatas=winddata,barmin=cbarmin,barmax=cbarmax,
                barlabel=barlabels[index]
            )
            index += 1
        if UTC_hour in range(16,result_hours-24+16,24): # 每过一天输出一次日均，无模拟时段(比目标时段长)的第一天和最后一天的均值
            index = 0
            WS_V_allmean += WS_V_daymean
            WS_U_allmean += WS_U_daymean
            WD_allmean += WD_daymean
            WS_V_daymean/=24
            WS_U_daymean/=24
            WD_daymean/=24
            for outdata in CMAQ_outdata:
                outdata_str = CMAQ_outdata_str[index]
                outdata_daymean = CMAQ_outdata_daymean[index]
                CMAQ_outdata_allmean[index]  += outdata_daymean
                os.makedirs(WCAS_output_dir+f"{outdata_str}/daymean/",exist_ok=True)
                pic_output_daymean = WCAS_output_dir+f"{outdata_str}/daymean/"
                outdata_daymean[:] /= 24
                cbarmax = getCbarmax(outdata_daymean[:])[0]
                cbarmin = getCbarmax(outdata_daymean[:])[1]
                barinterval = (cbarmax - cbarmin)/5
                date_yesterday = str(datetime.datetime.strptime(date_now.split(' ')[0], '%Y-%m-%d')) 
                # date_yesterday = str(datetime.datetime.strptime(date_now.split(' ')[0], '%Y-%m-%d')-datetime.timedelta(days=1)) # 每次是当前昨天的均值
                title = f"Daymean {CMAQ_outdata_str[index]} at {date_yesterday}"
                winddata_daymean = [XLONGD02[:], XLATD02[:],WS_V_daymean[:], WS_U_daymean[:] ,WD_daymean[:]]
                drawCoutourf(
                    outdata_daymean[:], LON[:, :], 
                    LAT[:, :], shpfiles, 
                    title, pic_output_daymean, cmaps[index], extent,barinterval,
                    'CMAQ',winddatas=winddata_daymean,barmin=cbarmin,barmax=cbarmax,
                    barlabel=barlabels[index]
                )
                CMAQ_outdata_daymean[index] = np.zeros((1, ROW, COL), dtype=np.float64) # 重置
                index +=1
            WS_V_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
            WS_U_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
            WD_daymean = np.zeros((1, ROWd02, COLd02), dtype=np.float64) 
        if UTC_hour == result_hours-24+16: # 最后一天，输出完整的模拟天的平均值
            index = 0
            WS_V_allmean/=result_hours-24
            WS_U_allmean/=result_hours-24
            WD_allmean/=result_hours-24
            for outdata in CMAQ_outdata:
                outdata_allmean = CMAQ_outdata_allmean[index]
                os.makedirs(WCAS_output_dir+f"{outdata_str}/",exist_ok=True)
                pic_output_allmean = WCAS_output_dir
                outdata_allmean[:] /= result_hours-24
                title = f"CMAQ simulation mean {CMAQ_outdata_str[index]}"
                winddata_allmean = [XLONGD02[:], XLATD02[:],WS_V_allmean[:], WS_U_allmean[:] ,WD_allmean[:]]
                cbarmax = getCbarmax(outdata_allmean[:])[0]
                cbarmin = getCbarmax(outdata_allmean[:])[1]
                barinterval = (cbarmax - cbarmin)/5
                drawCoutourf(
                    outdata_allmean[:], LON[:, :], 
                    LAT[:, :], shpfiles, 
                    title, pic_output_allmean, cmaps[index], extent,barinterval,
                    'CMAQ',winddatas=winddata_allmean,barmin=cbarmin,barmax=cbarmax,
                    barlabel=barlabels[index]
                )
                index +=1

#模型参数修改计算及流程控制
def terminal_exec(terminal: str, cmd: str):  # Control the terminal 
    """Execute cmd in the specific terminal

    Args:
        terminal (str): the terminal you would like to control
        cmd (str): command line, such as 'ls'
    """
    with open(terminal, "w") as fd:
        for c in "{}\n".format(cmd):
            fcntl.ioctl(fd, termios.TIOCSTI, c)
def get_unused_ttys(maxtty=15): # 返回未被使用的tty，顺序排列，然后就可以找到cmd打开的wsl新终端号，即第一个，maxtty指当前可能运行的最大终端数
    # 运行ps aux命令
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
    
    # 解析输出
    lines = result.stdout.split('\n')
    used_ttys = set()
    
    for line in lines[1:]:  # 跳过首行标题
        parts = line.split()
        if len(parts) > 6:  # 确保行具有足够的数据
            tty = parts[6]  # TTY值位于第7列
            if tty != '?':  # 忽略未知TTY
                used_ttys.add(tty)
    
    used_ttys_l = []
    for i in used_ttys:
        used_ttys_l.append(int(i.split('/')[1])) # 只获取号，便于顺序查找未被占用的一个，因为cmd启动会从低到高打开一个未使用的终端
    used_ttys_l.sort()

    all_settty = list(range(0,maxtty,1))
    unused_ttys_l = list(set(all_settty) - set(used_ttys_l))
    return unused_ttys_l
def check_process_status(process_name):
    try:
        # 执行命令获取进程列表
        output = subprocess.check_output(['ps', 'aux'])
        # 将输出转换为字符串
        output = output.decode('utf-8')
        # 按行分割输出
        lines = output.split('\n')
        # 遍历每行输出
        for line in lines:
            # 判断进程名称是否在输出中
            if process_name in line:
                return True
        return False
    except subprocess.CalledProcessError:
        # 命令执行失败
        return False
def modify_csh_variable(file_path, variable_name, new_value):
    # 读取原始文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 查找并修改对应的set变量行
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"set {variable_name} ="):
            # 构造新的变量声明行
            new_line = f"set {variable_name} = {new_value}\n"
            new_lines.append(new_line)
        elif line.strip().startswith(f"setenv {variable_name}"):
            new_line = f"setenv {variable_name} {new_value}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

#再分析资料下载
def downLoadFNLfiles(start_date, end_date, terminal):
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
            # baseURL = 'https://stratus.rda.ucar.edu/ds083.2/grib2/'
            baseURL = 'https://data.rda.ucar.edu/ds083.2/grib2/'
            filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
            for hour in hour_list:  # 获取月份
                filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib2'
                filelist.append(filename)
    opener = build_opener()
    for file in tqdm(filelist):
        #直接下载 在网络状况差时可能会banip且必须有VPN
        # ofile = os.path.basename(file)
        # # print("downloading " + ofile + " ... ")os.system('wget {0} -P {1}'.format(gfs_url, save_date_hour_path))
        # terminal_exec(terminal, "echo 'downloading" + ofile + "'...")
        # infile = opener.open(file)
        # outfile = open(ofile, "wb")
        # outfile.write(infile.read())
        # outfile.close()

        # 由于原先WCAS下载FNL的逻辑建立在'https://stratus.rda.ucar.edu/ds083.2/grib2/'下载指定时间段的fnl文件，
        # 时间段内有效的数据直接下载，无效的数据用空文件表示，空文件则说明GFS应以何时开始预报，而这个网站在2024.06.24左右失效，不再保留有效文件而全是空文件，则只能使用新url
        # 而新url在下载时，无效数据文件会下载失败，则为保留WCAS的执行逻辑不变，使用下述方法手动创建空文件
        
        #使用wget或aria2c下载
        if aria2c_download == True:
            if os.system(f'aria2c --file-allocation=none --check-certificate=false {file}') == 0:  # 路径中有预设给服务器的下载文件存放路径
                pass
            else:
                tempfile = os.path.basename(file)
                os.system(f'touch {tempfile}')
        else:
            if os.system(f'wget --no-check-certificate  {file} -P . > {terminal} 2>&1') == 0:  # 路径中有预设给服务器的下载文件存放路径
                pass
            else:
                tempfile = os.path.basename(file)
                os.system(f'touch {tempfile}')
            
        # terminal_exec(terminal, f'wget {file} -P .')       
def downLoadGFSfiles(
    save_path='.',
    target_date_in = '2024-03-16 00:00',
    after_hour = 8*3,
):
    """
    Athor: Xiang Dai
    """
    # 本代码用于GFS每日自动下载，如需下载指定时间，需要在下面网页中确认是否存在该日期，一般为最近10天，历史存档仅保留0.25度，
    # 网站为rda.ucar.edu的ds084.1数据集

    http = urllib3.PoolManager(cert_reqs='CERT_NONE') # 服务器端报错解决

    root_html = 'https://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/'
    save_path = save_path

    history_download = False
    target_date_download = True  # 如需下载单独时间，需要打开该标志
    target_date = datetime.datetime.strptime(target_date_in, "%Y-%m-%d %H:%M")  # 所需时间，UTC
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

            for f_hour_i in tqdm(range(after_hour//3)):
                gfs_url = root_html + 'gfs.{0}/{1}/atmos/gfs.t{1}z.pgrb2.1p00.f{2:03}'.format(download_date, hour_i, f_hour_i*3)
                if requests.get(gfs_url).status_code == 404:
                    return False
                file_name = gfs_url.split('/')[-1]
                if not os.path.exists(save_date_hour_path+file_name):
                    # os.system('{0} --url {1} --path {2} --retry_times {3}'.format(download_exe_path, gfs_url, save_date_hour_path+file_name, retry_times))
                    if aria2c_download == True:
                        os.system(f'aria2c -d {save_date_hour_path} {gfs_url} > {terminal}')
                    else:
                        os.system(f'wget --no-check-certificate {gfs_url} -P {save_date_hour_path} > {terminal} 2>&1')
                    time.sleep(1)

                url_file_size = get_file_size(gfs_url)
                if os.path.exists(save_date_hour_path+file_name) and (url_file_size != None) and (os.path.getsize(save_date_hour_path+file_name) != url_file_size):
                    # os.system('{0} --url {1} --path {2} --retry_times {3}'.format(download_exe_path, gfs_url, save_date_hour_path+file_name, retry_times))
                    if aria2c_download == True:
                        os.system(f'aria2c -d {save_date_hour_path} {gfs_url} > {terminal}')
                    else:
                        os.system(f'wget --no-check-certificate {gfs_url} -P {save_date_hour_path} > {terminal} 2>&1')
                    time.sleep(1)
                if (os.path.getsize(save_date_hour_path+'/'+file_name) != url_file_size):
                    os.remove(save_date_hour_path+file_name)

        #     print('{0}  {1}:00起未来{2}小时预测数据下载完成!'.format(download_date, hour_i, after_hour))
        # print('{0} 当日gfs所有时间段未来{1}小时预测数据下载完成!'.format(download_date, after_hour))
def downLoadFNLfiles_fromServer(start_date, end_date):
    hostname = remoteServer_hostname
    port = remoteServer_port  # SSH默认端口
    username = remoteServer_username
    password = remoteServer_password

    # 本地文件路径和远程服务器上的目标路径
    local_path = WCAS_dir
    remote_path = '/home/WCAS_NCARfilesdownload/' #默认的远程下载的目录，本地控制自动创建

    # paramiko连接服务器
    ssh = paramiko.SSHClient()# 创建SSH客户端
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())# 自动添加不安全的SSH主机密钥
    ssh.connect(hostname, port, username, password)# 连接到远程服务器

    #初始化远程下载目录
    stdin, stdout, stderr = ssh.exec_command(f'rm -rf {remote_path}') # 删除之前的远程目录
    stdout.channel.recv_exit_status()
    ssh.exec_command(f'mkdir {remote_path}') # 创建远程目录
    
    sftp = ssh.open_sftp() # 打开sftp功能
    sftp.put(f'{local_path}tools/downLoadFNLfiles_fromServer.py', f'{remote_path}downLoadFNLfiles_fromServer.py') # 传输下载脚本
    stdin, stdout, stderr = ssh.exec_command(f'conda activate WCAS_remote_download') # 启动远程服务器的conda，注意已经提前在远程服务器有所设置，这里没有全程自动设置
    stdout.channel.recv_exit_status()

    #远程调用脚本开始下载文件
    print(T.Now() + ' - 远程服务器开始下载FNL文件...')
    stdin, stdout, stderr = ssh.exec_command(f'python3 {remote_path}downLoadFNLfiles_fromServer.py {start_date} {end_date}')
    stdout.channel.recv_exit_status()
    # print(stdout.read())
    # print(stderr.read())

    #压缩所有fnl文件并传输回主服务器，再解压
    stdin, stdout, stderr = ssh.exec_command(f'tar -cvf {remote_path}fnlfiles.tar {remote_path}fnl*')
    stdout.channel.recv_exit_status()
    time.sleep(1)
    sftp.get(f'{remote_path}fnlfiles.tar', f'{local_path}fnlfiles.tar')
    os.system('tar -xf fnlfiles.tar --strip-components=2')
    os.system('rm -rf fnlfiles.tar')

    # 关闭连接
    sftp.close()
    ssh.close()
def downLoadGFSfiles_fromServer(save_path='.',target_date_in = '2024-03-16-00:00',after_hour = 8*3,):
    hostname = remoteServer_hostname
    port = remoteServer_port  # SSH默认端口
    username = remoteServer_username
    password = remoteServer_password

    # 本地文件路径和远程服务器上的目标路径
    local_path = WCAS_dir
    remote_path = '/home/WCAS_NCARfilesdownload/' #默认的远程下载的目录，本地控制自动创建

    # paramiko连接服务器
    ssh = paramiko.SSHClient()# 创建SSH客户端
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())# 自动添加不安全的SSH主机密钥
    ssh.connect(hostname, port, username, password)# 连接到远程服务器
    # ssh.exec_command(f'mkdir {remote_path}') # 创建远程目录  远程目录在FNL下载过程已经创建
    
    #初始化远程下载目录 当强制GFS下载时，才在这里进行目录初始化
    if ForceGFS == True:
        stdin, stdout, stderr = ssh.exec_command(f'rm -rf {remote_path}') # 删除之前的远程目录
        stdout.channel.recv_exit_status()
        ssh.exec_command(f'mkdir {remote_path}') # 创建远程目录

    sftp = ssh.open_sftp() # 打开sftp功能
    sftp.put(f'{local_path}tools/downLoadGFSfiles_fromServer.py', f'{remote_path}downLoadGFSfiles_fromServer.py') # 传输下载脚本
    stdin, stdout, stderr = ssh.exec_command(f'conda activate WCAS_remote_download') # 启动远程服务器的conda，注意已经提前在远程服务器有所设置，这里没有全程自动设置
    stdout.channel.recv_exit_status()

    #远程调用脚本开始下载文件
    # while True:
    print(T.Now() + ' - 远程服务器开始下载GFS文件...')
    stdin, stdout, stderr = ssh.exec_command(f'python3 {remote_path}downLoadGFSfiles_fromServer.py {save_path} {target_date_in} {after_hour}')
    # print(stdout.read())
    # print(stderr.read())
    stdout.channel.recv_exit_status()

    #压缩所有fnl文件并传输回主服务器，再解压
    stdin, stdout, stderr = ssh.exec_command(f'tar -cvf {remote_path}gfsfiles.tar {remote_path}gfs*')
    stdout.channel.recv_exit_status()
    time.sleep(1)
    print(T.Now() + ' - 将下载好的GFS文件从远程服务器打包传输...')
    sftp.get(f'{remote_path}gfsfiles.tar',f'{local_path}gfsfiles.tar')
    os.system('tar -xf gfsfiles.tar --strip-components=2') #不保留上级目录的解压
    os.system('rm -rf gfsfiles.tar')

    # 关闭连接
    sftp.close()
    ssh.close()

    
if __name__ == "__main__":
    T = Time()
    WCAS_start_time = time.time()

    WCAS_dir = os.path.abspath(__file__).split(os.path.basename(__file__))[0] # 常规执行
    # WCAS_dir = sys.argv[1] # pyinstall后的EXE方式执行：WCAS "$PWD/"

    
    print('####################################################')
    print('###         WCAS(WRF-CAMQ Auto Script)           ###')
    print('###                v2024.07.04                   ###')
    print('###          This Script is made by              ###')
    print('###           Yaohan Xian;Xiang Dai              ###')
    print('####################################################')

    print(T.Now() + ' - 读取namelist.WCAS')
    print(f'{WCAS_dir}namelist.WCAS')
    namelist_WCAS = f90nml.read(f"{WCAS_dir}namelist.WCAS")
    Mannualterminal = namelist_WCAS['control']['Mannualterminal']
    os.system(f'chmod u+x {WCAS_dir}namelists_cshfiles/*.csh') #所有WCAS脚本的执行权限

    if Mannualterminal == False: # 直接打开终端显示的模式，用于WSL环境，而Linux环境无法直接cmd打开窗口选择全部后台定向输出到文件中显示。
        terminal = f"/dev/pts/{str(get_unused_ttys()[0])}" # 被控制的，模型运行的的terminal 先检查最下一个tty号的空闲终端再打开
        os.system(f'cmd.exe /c start wt.exe wsl.exe -d Ubuntu-20.04') # 开启模型运行的终端
        time.sleep(2)  # 等待终端完全打开
    else:  # 手动设置终端情形，用于非WSL环境
        terminal = namelist_WCAS['control']['control_terminal']

    #路径
    WPS_dir = namelist_WCAS['dirs']['WPS_dir']
    WRF_dir = namelist_WCAS['dirs']['WRF_dir']
    CMAQ_dir = namelist_WCAS['dirs']['CMAQ_dir']
    premegan_dir = namelist_WCAS['dirs']['premegan_dir']
    premeganinput_dir = namelist_WCAS['dirs']['premeganinput_dir']
    premeganoutput_dir = namelist_WCAS['dirs']['premeganoutput_dir']
    MEGAN_dir = namelist_WCAS['dirs']['MEGAN_dir']
    MEIAT_dir = namelist_WCAS['dirs']['MEIAT_dir']
    MEIAT_python_dir = namelist_WCAS['dirs']['MEIAT_python_dir']
    #配置
    cores_CMAQ_col = namelist_WCAS['control']['cores_CMAQ_col']
    cores_CMAQ_row = namelist_WCAS['control']['cores_CMAQ_row']
    gridname = namelist_WCAS['control']['Gridname']
    CMAQsimtype = namelist_WCAS['control']['CMAQsimtype']
    regrid_dom = namelist_WCAS['control']['regrid_dom']
    regrid_D02andD03 = namelist_WCAS['control']['regrid_D02andD03']
    max_dom_manual = namelist_WCAS['control']['max_dom']
    CCTMversion = namelist_WCAS['control']['CCTMversion']
    BCICversion = namelist_WCAS['control']['BCICversion']
    compiler_str = namelist_WCAS['control']['compiler_str']
    aria2c_download = namelist_WCAS['control']['aria2c_download']
    remote_server_download = namelist_WCAS['control']['remote_server_download']
    if remote_server_download == True:
        remoteServer_hostname = namelist_WCAS['control']['remoteServer_hostname']
        remoteServer_port = namelist_WCAS['control']['remoteServer_port']
        remoteServer_username = namelist_WCAS['control']['remoteServer_username']
        remoteServer_password = namelist_WCAS['control']['remoteServer_password']
    ForceGFS = namelist_WCAS['control']['ForceGFS']
    MannualExtent = namelist_WCAS['control']['MannualExtent']
    Extent = namelist_WCAS['control']['Extent']
    testGeogrid = namelist_WCAS['control']['testGeogrid']
    dom_now_wind = namelist_WCAS['control']['dom_now_wind']
    ManualGrid = namelist_WCAS['control']['ManualGrid']
    runCMAQ = namelist_WCAS['control']['runCMAQ']
    runMEGAN = namelist_WCAS['control']['runMEGAN']
    runISAM = namelist_WCAS['control']['runISAM']
    runPostprocess = namelist_WCAS['control']['runPostprocess']
    CMAQcombine = namelist_WCAS['control']['CMAQcombine']
    cores_WRF = namelist_WCAS['control']['cores_WRF']
    cores_win = namelist_WCAS['control']['cores_win']
    MEIAT_Linux = namelist_WCAS['control']['MEIAT_Linux']
    MEIAT_Linux_dir = namelist_WCAS['control']['MEIAT_Linux_dir']
    #ISAM
    if_makemask = namelist_WCAS['ISAMcontrol']['if_makemask']
    mask_shpfilenames = namelist_WCAS['ISAMcontrol']['mask_shpfilenames']
    mask_varnames = namelist_WCAS['ISAMcontrol']['mask_varnames']
    if isinstance(mask_varnames, list):
        mask_varnames_l = []
        for i in mask_varnames:
            mask_varnames_l.append(i)
    else:
        mask_varnames_l = []
        mask_varnames_l.append(mask_varnames)
        
    # 目标模拟结果时段
    start_date = namelist_WCAS['date']['start_date']
    end_date = namelist_WCAS['date']['end_date']

    # 计算各个模块的时间变量
    start_date_WRF = str(datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=2)).split(' ')[0]  # 提前2天以移除世界时偏移和MCIP限制,datetime增减时间要自动加一个时分秒，通过split去除
    end_date_WRF = str(datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=2)).split(' ')[0]  # 延后2天以移除世界时偏移和MCIP限制
    run_days_WRF = (datetime.datetime.strptime(end_date_WRF, '%Y-%m-%d') - datetime.datetime.strptime(start_date_WRF, '%Y-%m-%d')).days # WRFnamelist里第一个参数rundays,
    start_date_MCIP = str(datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=1)).split(' ')[0]  # MCIP开始不能是WRF起始，相对输入起始时间-1，WRF是-2
    end_date_MCIP = str(datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)).split(' ')[0] ## MCIP结束比WRF早一天
    run_days_MCIP = (datetime.datetime.strptime(end_date_MCIP, '%Y-%m-%d') - datetime.datetime.strptime(start_date_MCIP, '%Y-%m-%d')).days # MCIP文件生成了几个，用于判断是否正常
    ICON_APPL = start_date_MCIP.replace('-', '')
    emis_start_date = start_date_MCIP  # 排放清单开始时间，MEGAN，MEIAT都使用这2个时间段
    emis_end_date = str(datetime.datetime.strptime(end_date_MCIP, '%Y-%m-%d') - datetime.timedelta(days=1)).split(' ')[0] # 排放清单结束时间比mcip早一天
    cctm_start_date = emis_start_date # CCTM模拟时间段和排放清单时间段一致
    cctm_end_date = emis_end_date

    # print(T.Now() + ' - 下载FNL再分析资料')
    # if not os.path.exists(f'{WCAS_dir}{gridname}fnlfiles/'):
    #     downLoadFNLfiles(start_date_WRF, end_date_WRF, terminal)
    #     os.system(f'mkdir {WCAS_dir}{gridname}fnlfiles/')
    #     os.system(f'mv {WCAS_dir}fnl* {WCAS_dir}{gridname}fnlfiles/')
    #     print(T.Now() + ' - 下载FNL再分析资料下载完成')
    # else:
    #     print(T.Now() + ' - 此次模拟的FNL再分析资料已经下载')

    print(T.Now() + ' - 下载FNL/GFS再分析资料')
    if not os.path.exists(f'{WCAS_dir}simulations/{gridname}/fnlfiles/'):
        if ForceGFS == True:
            GFS = 1
            print(T.Now() + ' - 已设置强制全为GFS预报场资料')
            days_will_GFS = datetime.datetime.strptime(end_date_WRF, '%Y-%m-%d')-datetime.datetime.strptime(start_date_WRF, '%Y-%m-%d')
            print(T.Now() + f"需要的GFS预报天数：{days_will_GFS.days} 基本小时：{21}")
            # if(int(days_will_GFS.days) >= 14):
                # print(T.Now() + f"预报天数超过GFS目前预报范围，请重新输入模拟时段！")
                # sys.exit()
            
            hours_will_GFS = days_will_GFS.days*24+21
            print(T.Now() + ' - 下载GFS预报场资料')
            while(True):
                # 服务器远程预报暂时没有自适应功能
                # GFS_start_str_for_sever_1 = str(error_date)[:-3].split(' ')
                # GFS_start_str_for_sever_2 = f'{GFS_start_str_for_sever_1[0]}-{GFS_start_str_for_sever_1[1]}' # 去掉空格而用-的格式
                downLoadGFSfiles_fromServer('.',start_date_WRF+'-00:00',hours_will_GFS)
                break
                

            print(T.Now() + ' - GFS预报场资料下载完成')
        else:
            print(T.Now() + ' - 下载FNL再分析资料')
            if remote_server_download == True:
                downLoadFNLfiles_fromServer(start_date_WRF, end_date_WRF)
            else:
                downLoadFNLfiles(start_date_WRF, end_date_WRF, terminal)
            print(T.Now() + ' - FNL再分析资料下载完成')
            FNLfileslist = [i for i in os.listdir('.') if i.split('_')[0] == 'fnl']
            FNLfileslist.sort() # 有序的时间
            print(T.Now() + ' - 检查FNL资料完整性')
            GFS = 0
            for fnl in FNLfileslist: 
                if os.path.getsize(f"./{fnl}") < 100: # 文件无效
                    error_date_name = fnl.split('_')[1]
                    error_date = datetime.datetime.strptime(error_date_name, '%Y%m%d')
                    GFS = 1 
                    break # 无效的第一天开始后面都没有数据
            if GFS == 1:
                print(T.Now() + f" {error_date} 后无完整FNL资料！将当天之后的切换为预报场GFS,且WPS以10800interval进行ungrib")
                
                days_will_GFS = datetime.datetime.strptime(end_date_WRF, '%Y-%m-%d')-error_date
                print(T.Now() + f"需要的GFS预报天数：{days_will_GFS.days} 基本小时：{21}")
                # if(int(days_will_GFS.days) >= 14):
                #     print(T.Now() + f"预报天数超过GFS目前预报范围，请重新输入模拟时段！")
                #     sys.exit()
                
                hours_will_GFS = days_will_GFS.days*24+21
                print(T.Now() + ' - 下载GFS预报场资料')
                while(True):
                    if remote_server_download == True:
                        # 服务器远程预报暂时没有自适应功能
                        GFS_start_str_for_sever_1 = str(error_date)[:-3].split(' ')
                        GFS_start_str_for_sever_2 = f'{GFS_start_str_for_sever_1[0]}-{GFS_start_str_for_sever_1[1]}' # 去掉空格而用-的格式
                        downLoadGFSfiles_fromServer('.',GFS_start_str_for_sever_2,hours_will_GFS)
                        break
                    else:
                        if downLoadGFSfiles('.',str(error_date)[:-3],hours_will_GFS) == False:
                            print(T.Now() + f" {str(error_date)}开始的GFS预报场资料还未上传，GFS模拟起始时间提前1天")
                            hours_will_GFS+=24
                            error_date = datetime.datetime.strptime(str(error_date).split(' ')[0], '%Y-%m-%d') - datetime.timedelta(days=1)

                            days_will_GFS = datetime.datetime.strptime(end_date_WRF, '%Y-%m-%d')-error_date
                            print(T.Now() + f"需要的GFS预报天数：{days_will_GFS.days}")
                            if int(days_will_GFS.days) == 0:
                                print(T.Now() + f"预报起始时间应在现实时间前一天，请重新输入模拟时段！")
                                sys.exit()
                        else:
                            break

                print(T.Now() + ' - GFS预报场资料下载完成')

        os.system(f'mkdir {WCAS_dir}simulations/{gridname}')
        os.system(f'mkdir {WCAS_dir}simulations/{gridname}/fnlfiles')
        os.system(f'mv {WCAS_dir}fnl* {WCAS_dir}simulations/{gridname}/fnlfiles/')
        time.sleep(1.5)
        if GFS == 1:
            os.system(f'mv {WCAS_dir}gfs.* {WCAS_dir}simulations/{gridname}/fnlfiles/')
            time.sleep(3)
            allFNLGFSlist = os.listdir(f"{WCAS_dir}simulations/{gridname}/fnlfiles/") # 删除无效文件
            RMfiles1,RMfiles2 = [],[]
            for i in allFNLGFSlist:
                if os.path.getsize(f"{WCAS_dir}simulations/{gridname}/fnlfiles/{i}") < 100: # 删除无效文件
                    RMfiles1.append(i) # 先获取删除的无效文件，在各个删除
                if i[-2:] in ['.1','.2','.3']: # 删除框架文件(ungrib能运行但值为空)
                    RMfiles2.append(i)
            if RMfiles1 != []:
                for f in RMfiles1:
                    os.system(f'rm -f {WCAS_dir}simulations/{gridname}/fnlfiles/{f}')
            if RMfiles2 != []:
                for f in RMfiles2:
                    os.system(f'rm -f {WCAS_dir}simulations/{gridname}/fnlfiles/{f}')
            error_date_del = str(error_date).split(' ')[0]
            error_date_del2 = f"{error_date_del.split('-')[0]}{error_date_del.split('-')[1]}{error_date_del.split('-')[2]}"
            os.system(f'rm -f {WCAS_dir}simulations/{gridname}/fnlfiles/fnl_{error_date_del2}*') # 有效但属于不完整天的fnl也删除
    else:
        filessss = os.listdir(f'{WCAS_dir}simulations/{gridname}/fnlfiles/') # 检测是否有GFS资料，如果有需要改变interval
        for i in filessss:
            if i.split('.')[0] == 'gfs':
                GFS = 1
            else: GFS = 0
        print(T.Now() + ' - 此次模拟的FNL/GFS再分析资料已经下载')
    terminal_exec(terminal, f'mkdir {WCAS_dir}simulations/{gridname}/flagfiles') # flags文件夹，如果中途终端，可跳过已经模拟完成的长时间过程(WRF,MEIAT,CCTM)

    
    print(T.Now() + ' - 计算嵌套区域数据')
    if ManualGrid == False: # 基于经纬度计算grid模式：
        grid_dict = wrf_grid_calculate(lon_lat_limit={'d01': [[namelist_WCAS['grid']['lat_min_d01'], namelist_WCAS['grid']['lat_max_d01']],
                                              [namelist_WCAS['grid']['lon_min_d01'], namelist_WCAS['grid']['lon_max_d01']]],
                                      'd02': [[namelist_WCAS['grid']['lat_min_d02'], namelist_WCAS['grid']['lat_max_d02']],
                                              [namelist_WCAS['grid']['lon_min_d02'], namelist_WCAS['grid']['lon_max_d02']]],  #
                                      'd03': [[namelist_WCAS['grid']['lat_min_d03'], namelist_WCAS['grid']['lat_max_d03']],
                                              [namelist_WCAS['grid']['lon_min_d03'], namelist_WCAS['grid']['lon_max_d03']]]},
                                       res_d01=namelist_WCAS['grid']['res_d01'],
                                       true_lat1 = namelist_WCAS['grid']['std_lat1_d01'],
                                       true_lat2 = namelist_WCAS['grid']['std_lat2_d01']
                       )
        resd01 = namelist_WCAS['grid']['res_d01']
    else: # 基于grid参数的模式，主要是为了重新模拟之前已有的嵌套区域
        grid_dict = {}
        
        grid_dict.update({'e_we':namelist_WCAS['manualgrid']['e_we']})
        grid_dict.update({'e_sn':namelist_WCAS['manualgrid']['e_sn']})
        grid_dict.update({'i_start': namelist_WCAS['manualgrid']['i_parent_start']})
        grid_dict.update({'j_start': namelist_WCAS['manualgrid']['j_parent_start']})
        grid_dict.update({'ref_lon': namelist_WCAS['manualgrid']['ref_lon']})
        grid_dict.update({'ref_lat': namelist_WCAS['manualgrid']['ref_lat']})
        grid_dict.update({'truelat1': namelist_WCAS['manualgrid']['truelat1']})
        grid_dict.update({'truelat2': namelist_WCAS['manualgrid']['truelat2']})
        grid_dict.update({'stand_lon': namelist_WCAS['manualgrid']['stand_lon']})
        resd01 = namelist_WCAS['manualgrid']['dx']
    with open(f'{WCAS_dir}simulations/{gridname}/geogridinfo.txt', 'w') as file:
        print(grid_dict, file=file)
    print(T.Now() + ' - wrf_lambert_grid_caculating计算完成')    

    print(T.Now() + ' - 将模拟时间段和嵌套范围数据写入namelist.wps')
    if max_dom_manual >= 2: i_parent_start_2 = grid_dict['i_start'][1]
    if max_dom_manual >= 3: i_parent_start_3 = grid_dict['i_start'][2]
    if max_dom_manual >= 2: j_parent_start_2 = grid_dict['j_start'][1]
    if max_dom_manual >= 3: j_parent_start_3 = grid_dict['j_start'][2]
    e_we_1 = grid_dict['e_we'][0]
    if max_dom_manual >= 2: e_we_2 = grid_dict['e_we'][1]
    if max_dom_manual >= 3: e_we_3 = grid_dict['e_we'][2]
    e_sn_1 = grid_dict['e_sn'][0]
    if max_dom_manual >= 2: e_sn_2 = grid_dict['e_sn'][1]
    if max_dom_manual >= 3: e_sn_3 = grid_dict['e_sn'][2]
    ref_lon = grid_dict['ref_lon']
    ref_lat = grid_dict['ref_lat']
    stand_lon = grid_dict['stand_lon']
    truelat1 = grid_dict['truelat1']
    truelat2 = grid_dict['truelat2']

    
    namelist_wps = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.wps.temp") #
    hour = '_00:00:00'  # 暂时未考虑非0时时段
    namelist_wps['share']['max_dom'] = max_dom_manual
    namelist_wps['share']['start_date'] = [start_date_WRF + hour, start_date_WRF + hour, start_date_WRF + hour]
    namelist_wps['share']['end_date'] = [end_date_WRF + hour, end_date_WRF + hour, end_date_WRF + hour]
    if max_dom_manual == 2:
        namelist_wps['geogrid']['parent_id'] = [1, 1]
        namelist_wps['geogrid']['parent_grid_ratio'] = [1, 3]
        namelist_wps['geogrid']['i_parent_start'] = [1, i_parent_start_2]
        namelist_wps['geogrid']['j_parent_start'] = [1, j_parent_start_2]
        namelist_wps['geogrid']['e_we'] = [e_we_1, e_we_2]
        namelist_wps['geogrid']['e_sn'] = [e_sn_1, e_sn_2]
    if max_dom_manual == 3:
        namelist_wps['geogrid']['parent_id'] = [1, 1, 2]
        namelist_wps['geogrid']['parent_grid_ratio'] = [1, 3, 3]
        namelist_wps['geogrid']['i_parent_start'] = [1, i_parent_start_2, i_parent_start_3]
        namelist_wps['geogrid']['j_parent_start'] = [1, j_parent_start_2, j_parent_start_3]
        namelist_wps['geogrid']['e_we'] = [e_we_1, e_we_2, e_we_3]
        namelist_wps['geogrid']['e_sn'] = [e_sn_1, e_sn_2, e_sn_3]
    namelist_wps['geogrid']['ref_lat'] = [ref_lat]
    namelist_wps['geogrid']['ref_lon'] = [ref_lon]
    namelist_wps['geogrid']['truelat1'] = [truelat1]
    namelist_wps['geogrid']['truelat2'] = [truelat2]
    namelist_wps['geogrid']['stand_lon'] = [stand_lon]
    namelist_wps['geogrid']['dx'] = resd01
    namelist_wps['geogrid']['dy'] = resd01
    if GFS == 1: namelist_wps['share']['interval_seconds'] = 10800
    else: namelist_wps['share']['interval_seconds'] = 21600
    namelist_wps.write(f"{WPS_dir}namelist.wps", force=True)
    print(T.Now() + ' - namelist.wps写入完成')


    
    print(T.Now() + ' - 开始运行WPS')
    # ========================================WPS======================================== 注释此部分可跳过,用于调试
    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/WPS.ok") == False:
        print(T.Now() + ' - 删除WPS与WRF的历史临时文件...')
        terminal_exec(terminal, f"cd {WPS_dir}")
        terminal_exec(terminal, "rm -f met_em*")
        terminal_exec(terminal, "rm -f FILE*")
        terminal_exec(terminal, "rm -f GRIBFILE*")
        terminal_exec(terminal, f"cd {WRF_dir}run")
        terminal_exec(terminal, "rm -f met_em*")
        terminal_exec(terminal, "rm -f wrfbdy*")
        terminal_exec(terminal, "rm -f wrfinput*")
        terminal_exec(terminal, "rm -f rsl.*")
        print(T.Now() + ' - WPS与WRF的历史临时文件清空完成')

        terminal_exec(terminal, f"mkdir {WCAS_dir}simulations/{gridname}/WPSoutput")
        terminal_exec(terminal, f"cd {WPS_dir}")
        if not testGeogrid:
            print(T.Now() + ' - ' + '    开始运行geogrid.exe...')
            terminal_exec(terminal, "./geogrid/geogrid.exe")
            time.sleep(3)
            while True:
                time.sleep(1)
                if not check_process_status('geogrid.exe'):  # 检查进程是否结束
                    break
            print(T.Now() + ' - ' + '    geogrid.exe运行完成')
            terminal_exec(terminal, f"cp geo_em* {WCAS_dir}simulations/{gridname}/WPSoutput")
        else:
            print(T.Now() + ' - ' + '    开始运行geogrid.exe...')
            terminal_exec(terminal, "./geogrid/geogrid.exe")
            time.sleep(3)
            while True:
                time.sleep(1)
                if not check_process_status('geogrid.exe'):  # 检查进程是否结束
                    break
            print(T.Now() + ' - ' + '    geogrid.exe运行完成')
            terminal_exec(terminal, f"cp geo_em* {WCAS_dir}simulations/{gridname}/WPSoutput")
            drawGeogrid(   
                geo_em_files_dir = f"{WCAS_dir}simulations/{gridname}/WPSoutput/",  
                geogrid_picout_dir = f"{WCAS_dir}simulations/{gridname}/"
            )
            print(T.Now() + ' - ' + f'    geogrid.exe嵌套层参数和图像已经输出到：{WCAS_dir}simulations/{gridname}/')
            while True:
                choice = input("输入: 修改namelist.WCAS的GRID后重新运行(R) 完成geogrid测试继续模拟的下一步(Y)")
                if choice == 'Y':
                    break
                if choice == 'R':
                    namelist_WCAS = f90nml.read(f"{WCAS_dir}namelist.WCAS") # 再次读取
                    if ManualGrid == False: # 基于经纬度计算grid模式：
                        grid_dict = wrf_grid_calculate(lon_lat_limit={'d01': [[namelist_WCAS['grid']['lat_min_d01'], namelist_WCAS['grid']['lat_max_d01']],
                                                            [namelist_WCAS['grid']['lon_min_d01'], namelist_WCAS['grid']['lon_max_d01']]],
                                                    'd02': [[namelist_WCAS['grid']['lat_min_d02'], namelist_WCAS['grid']['lat_max_d02']],
                                                            [namelist_WCAS['grid']['lon_min_d02'], namelist_WCAS['grid']['lon_max_d02']]],  #
                                                    'd03': [[namelist_WCAS['grid']['lat_min_d03'], namelist_WCAS['grid']['lat_max_d03']],
                                                            [namelist_WCAS['grid']['lon_min_d03'], namelist_WCAS['grid']['lon_max_d03']]]},
                                                    res_d01=namelist_WCAS['grid']['res_d01'],
                                                    true_lat1 = namelist_WCAS['grid']['std_lat1_d01'],
                                                    true_lat2 = namelist_WCAS['grid']['std_lat2_d01']
                                    )
                        resd01 = namelist_WCAS['grid']['res_d01']
                    else: # 基于grid参数的模式，主要是为了重新模拟之前已有的嵌套区域
                        grid_dict = {}
                        grid_dict.update({'e_we':namelist_WCAS['manualgrid']['e_we']})
                        grid_dict.update({'e_sn':namelist_WCAS['manualgrid']['e_sn']})
                        grid_dict.update({'i_start': namelist_WCAS['manualgrid']['i_parent_start']})
                        grid_dict.update({'j_start': namelist_WCAS['manualgrid']['j_parent_start']})
                        grid_dict.update({'ref_lon': namelist_WCAS['manualgrid']['ref_lon']})
                        grid_dict.update({'ref_lat': namelist_WCAS['manualgrid']['ref_lat']})
                        grid_dict.update({'truelat1': namelist_WCAS['manualgrid']['truelat1']})
                        grid_dict.update({'truelat2': namelist_WCAS['manualgrid']['truelat2']})
                        grid_dict.update({'stand_lon': namelist_WCAS['manualgrid']['stand_lon']})
                        resd01 = namelist_WCAS['manualgrid']['dx']
                    with open(f'{WCAS_dir}simulations/{gridname}/geogridinfo.txt', 'w') as file:
                        print(grid_dict, file=file)
                    print(T.Now() + ' - 将模拟时间段和嵌套范围数据写入namelist.wps')
                    if max_dom_manual >= 2: i_parent_start_2 = grid_dict['i_start'][1]
                    if max_dom_manual >= 3: i_parent_start_3 = grid_dict['i_start'][2]
                    if max_dom_manual >= 2: j_parent_start_2 = grid_dict['j_start'][1]
                    if max_dom_manual >= 3: j_parent_start_3 = grid_dict['j_start'][2]
                    e_we_1 = grid_dict['e_we'][0]
                    if max_dom_manual >= 2: e_we_2 = grid_dict['e_we'][1]
                    if max_dom_manual >= 3: e_we_3 = grid_dict['e_we'][2]
                    e_sn_1 = grid_dict['e_sn'][0]
                    if max_dom_manual >= 2: e_sn_2 = grid_dict['e_sn'][1]
                    if max_dom_manual >= 3: e_sn_3 = grid_dict['e_sn'][2]
                    ref_lon = grid_dict['ref_lon']
                    ref_lat = grid_dict['ref_lat']
                    stand_lon = grid_dict['stand_lon']
                    truelat1 = grid_dict['truelat1']
                    truelat2 = grid_dict['truelat2']
                    namelist_wps = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.wps.temp") #
                    hour = '_00:00:00'  # 暂时未考虑非0时时段
                    namelist_wps['share']['max_dom'] = max_dom_manual
                    namelist_wps['share']['start_date'] = [start_date_WRF + hour, start_date_WRF + hour, start_date_WRF + hour]
                    namelist_wps['share']['end_date'] = [end_date_WRF + hour, end_date_WRF + hour, end_date_WRF + hour]
                    if max_dom_manual == 2:
                        namelist_wps['geogrid']['parent_id'] = [1, 1]
                        namelist_wps['geogrid']['parent_grid_ratio'] = [1, 3]
                        namelist_wps['geogrid']['i_parent_start'] = [1, i_parent_start_2]
                        namelist_wps['geogrid']['j_parent_start'] = [1, j_parent_start_2]
                        namelist_wps['geogrid']['e_we'] = [e_we_1, e_we_2]
                        namelist_wps['geogrid']['e_sn'] = [e_sn_1, e_sn_2]
                    if max_dom_manual == 3:
                        namelist_wps['geogrid']['parent_id'] = [1, 1, 2]
                        namelist_wps['geogrid']['parent_grid_ratio'] = [1, 3, 3]
                        namelist_wps['geogrid']['i_parent_start'] = [1, i_parent_start_2, i_parent_start_3]
                        namelist_wps['geogrid']['j_parent_start'] = [1, j_parent_start_2, j_parent_start_3]
                        namelist_wps['geogrid']['e_we'] = [e_we_1, e_we_2, e_we_3]
                        namelist_wps['geogrid']['e_sn'] = [e_sn_1, e_sn_2, e_sn_3]
                    namelist_wps['geogrid']['ref_lat'] = [ref_lat]
                    namelist_wps['geogrid']['ref_lon'] = [ref_lon]
                    namelist_wps['geogrid']['truelat1'] = [truelat1]
                    namelist_wps['geogrid']['truelat2'] = [truelat2]
                    namelist_wps['geogrid']['stand_lon'] = [stand_lon]
                    namelist_wps['geogrid']['dx'] = resd01
                    namelist_wps['geogrid']['dy'] = resd01
                    if GFS == 1: namelist_wps['share']['interval_seconds'] = 10800
                    else: namelist_wps['share']['interval_seconds'] = 21600
                    namelist_wps.write(f"{WPS_dir}namelist.wps", force=True)
                    print(T.Now() + ' - namelist.wps写入完成')
                    print(T.Now() + ' - ' + '    开始运行geogrid.exe...')
                    terminal_exec(terminal, "./geogrid/geogrid.exe")
                    time.sleep(3)
                    while True:
                        time.sleep(1)
                        if not check_process_status('geogrid.exe'):  # 检查进程是否结束
                            break
                    print(T.Now() + ' - ' + '    geogrid.exe运行完成')
                    terminal_exec(terminal, f"rm -f {WCAS_dir}simulations/{gridname}/WPSoutput/geo_em*")
                    time.sleep(1)
                    terminal_exec(terminal, f"cp geo_em* {WCAS_dir}simulations/{gridname}/WPSoutput")
                    time.sleep(1)
                    drawGeogrid(   
                        geo_em_files_dir = f"{WCAS_dir}simulations/{gridname}/WPSoutput/",  
                        geogrid_picout_dir = f"{WCAS_dir}simulations/{gridname}/"
                    )
                    print(T.Now() + ' - ' + f'    geogrid.exe嵌套层参数和图像已经输出到：{WCAS_dir}simulations/{gridname}/')
                    
            
        print(T.Now() + ' -    开始运行ungrib.exe...')
        terminal_exec(terminal, f"ln -sf {WPS_dir}ungrib/Variable_Tables/Vtable.GFS Vtable")
        terminal_exec(terminal, f"./link_grib.csh {WCAS_dir}simulations/{gridname}/fnlfiles/*")
        terminal_exec(terminal, "./ungrib/ungrib.exe")
        time.sleep(3)  # 防止进程还没开始就进行检查
        while True:
            time.sleep(1)
            if check_process_status('ungrib.exe') == False:  # 检查进程是否结束
                break
        print(T.Now() + ' - ' + '    ungrib.exe运行完成')
        print(T.Now() + ' - ' + '    开始运行metgrid.exe...')
        terminal_exec(terminal, "./metgrid/metgrid.exe")
        time.sleep(3)
        while True:
            time.sleep(1)
            if not check_process_status('metgrid.exe'):  # 检查进程是否结束
                break
        print(T.Now() + ' - ' + '    metgrid.exe运行完成')
    # ========================================WPS======================================== 注释此部分可跳过,用于调试

    print(T.Now() + ' - ' + 'WPS运行完成')
    terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/WPS.ok")
    

    print(T.Now() + ' - '+'将模拟时间段和嵌套范围数据写入namelist.input')
    namelist_input = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.input.temp")
    run_days = (datetime.datetime.strptime(end_date_WRF, '%Y-%m-%d') - datetime.datetime.strptime(start_date_WRF, '%Y-%m-%d')).days
    namelist_input['time_control']['run_days'] = run_days_WRF
    if ManualGrid == True:
        namelist_input['domains']['max_dom'] = max_dom_manual
        namelist_input['time_control']['start_year'] = [int(start_date_WRF.split('-')[0])]*max_dom_manual
        namelist_input['time_control']['start_month'] = [int(start_date_WRF.split('-')[1])]*max_dom_manual
        namelist_input['time_control']['start_day'] = [int(start_date_WRF.split('-')[2])]*max_dom_manual
        namelist_input['time_control']['end_year'] = [int(end_date_WRF.split('-')[0])]*max_dom_manual
        namelist_input['time_control']['end_month'] = [int(end_date_WRF.split('-')[1])]*max_dom_manual
        namelist_input['time_control']['end_day'] = [int(end_date_WRF.split('-')[2])]*max_dom_manual
    else:
        namelist_input['time_control']['start_year'] = [int(start_date_WRF.split('-')[0])]*3
        namelist_input['time_control']['start_month'] = [int(start_date_WRF.split('-')[1])]*3
        namelist_input['time_control']['start_day'] = [int(start_date_WRF.split('-')[2])]*3
        namelist_input['time_control']['end_year'] = [int(end_date_WRF.split('-')[0])]*3
        namelist_input['time_control']['end_month'] = [int(end_date_WRF.split('-')[1])]*3
        namelist_input['time_control']['end_day'] = [int(end_date_WRF.split('-')[2])]*3
    if max_dom_manual == 1:
        namelist_input['domains']['e_we'] = [e_we_1]
        namelist_input['domains']['e_sn'] = [e_sn_1]
        namelist_input['domains']['i_parent_start'] = [1]
        namelist_input['domains']['j_parent_start'] = [1]
    if max_dom_manual == 2:
        namelist_input['domains']['e_we'] = [e_we_1, e_we_2]
        namelist_input['domains']['e_sn'] = [e_sn_1, e_sn_2]
        namelist_input['domains']['i_parent_start'] = [1, i_parent_start_2]
        namelist_input['domains']['j_parent_start'] = [1, j_parent_start_2]
    if max_dom_manual == 3:
        namelist_input['domains']['e_we'] = [e_we_1, e_we_2, e_we_3]
        namelist_input['domains']['e_sn'] = [e_sn_1, e_sn_2, e_sn_3]
        namelist_input['domains']['i_parent_start'] = [1, i_parent_start_2, i_parent_start_3]
        namelist_input['domains']['j_parent_start'] = [1, j_parent_start_2, j_parent_start_3]
    namelist_input['domains']['dx'] = [resd01, int(resd01/3), int(resd01/3/3)]
    namelist_input['domains']['dy'] = [resd01, int(resd01 / 3),int(resd01/ 3 / 3)]
    if GFS == 1: namelist_input['time_control']['interval_seconds'] = 10800
    else: namelist_input['time_control']['interval_seconds'] = 21600
    namelist_input.write(f"{WRF_dir}run/namelist.input", force=True)
    print(T.Now() + ' - '+'namelist.input生成成功 ')
    print(T.Now() + ' - '+'开始运行WRF...')
    terminal_exec(terminal, f"cd {WRF_dir}run")
    terminal_exec(terminal, f"ln -sf {WPS_dir}met_em.d0* {WRF_dir}run")
    terminal_exec(terminal, "ulimit -d unlimited")
    terminal_exec(terminal, "ulimit -s unlimited")
    terminal_exec(terminal, "ulimit -c unlimited")
    # ========================================WRF======================================== 注释此部分可跳过,用于调试
    
    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/WRF.ok") == False:
        terminal_exec(terminal, "./real.exe")
        time.sleep(5)
        while (True):
            time.sleep(1)
            if check_process_status('real.exe') == False:  # 检查进程是否结束
                break
        terminal_exec(terminal, F"mpiexec -n {cores_WRF} ./wrf.exe ")
        time.sleep(2)
        if Mannualterminal == False: 
            terminal_WRFinfo = f"/dev/pts/{str(get_unused_ttys()[0])}" # 被控制的，模型运行的的terminal
            os.system(f'cmd.exe /c start wt.exe wsl.exe -d Ubuntu-20.04') # 查看WRF进度的终端
            time.sleep(5)
            terminal_exec(terminal_WRFinfo, f"tail -f {WRF_dir}run/rsl.out.0000")
        time.sleep(10)
        while (True):
            time.sleep(1)
            if check_process_status('wrf.exe') == False:  # 检查进程是否结束
                break
        terminal_exec(terminal, "mkdir WRFoutput")
        time.sleep(0.5)
        terminal_exec(terminal, "mv wrfi* wrfo* wrfr* wrfbdy* WRFoutput")
        time.sleep(7) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
        if os.path.exists(f"{WRF_dir}run/WRFoutput/wrfout_d01_{end_date_WRF.split('-')[0]}-{end_date_WRF.split('-')[1]}-{end_date_WRF.split('-')[2]}_00:00:00") == False:
            print(T.Now() + ' - '+' 错误!WRF结果未能正常输出，检查namelist.input.temp以及错误报告!')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/WRF.error.FNLGFSmissing")  # 为服务器设置的重启策略检查文件，已经服务器上人为一般是资料下载不完整导致的错误
            sys.exit()
    # ========================================WRF======================================== 注释此部分可跳过,用于调试
    print(T.Now() + ' - '+'WRF运行完成！')
    terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/WRF.ok")
    
    
    # ===============================================================================================================================
    # profile: 直接以清洁空气边界场的方式模拟最里层
    # ===============================================================================================================================
    if runCMAQ:    
        CMAQdata_dir = f"{WCAS_dir}simulations/{gridname}/"
        if CMAQsimtype == 'profile': # 直接profile模拟，则直接模拟第三层
            print(T.Now() + ' - ' + f'开始准备CMAQ的模拟 - 模拟方式 - {CMAQsimtype}')
        
            print(T.Now() + ' - ' +'    开始运行MCIP')
            MCIP_GridName = gridname + '_d0'+str(regrid_dom)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','CMAQ_HOME',CMAQ_dir)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','start_time',start_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','end_time',end_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','domain',regrid_dom) # 最里层dom
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','GridName',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','DataPath',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InMetDir',f"{WRF_dir}run/WRFoutput")
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InGeoDir',f"{WCAS_dir}simulations/{gridname}/WPSoutput")
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','WRF_LC_REF_LAT',ref_lat)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/mcip/scripts/") #
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/mcip/scripts/run_mcip_daybyday_profile_WCAS.csh")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MCIP_profile.ok") == False:
                terminal_exec(terminal, f"./run_mcip_daybyday_profile_WCAS.csh {compiler_str}")
                time.sleep(2)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/mcip/scripts/mcipcsh.ok") == True: # 由于daybyday运行脚本在循环时exe进程不连续，可能导致终端，这里便采用flag文件进行状态判断而不是检测exe
                        break
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip")) < 2+run_days_MCIP*9 and os.path.exists(f"{CMAQdata_dir}{MCIP_GridName}/mcip") == True: # 文件生成数量是否正常(2个单独的和9类)，第二个条件是防止找不到路径报错
                    print(T.Now() + ' - '+' 错误! MCIP结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
            print(T.Now() + ' - ' +'    MCIP运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MCIP_profile.ok")
        
            print(T.Now() + ' - ' +'    开始运行BCON.....')
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','BCTYPE',CMAQsimtype)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','start_time',start_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','end_time',end_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/bcon/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/bcon/scripts/run_bcon_daybyday_profile_WCAS.csh")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/BCON_profile.ok") == False:
                terminal_exec(terminal, "./run_bcon_daybyday_profile_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/bcon/scripts/bconcsh.ok") == True:
                        break
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/bcon")) < run_days_MCIP*1: # 文件生成数量是否正常
                    print(T.Now() + ' - '+' 错误! BCON结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
            print(T.Now() + ' - ' +'    BCON运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/BCON_profile.ok")
        
            print(T.Now() + ' - ' +'    开始运行ICON.....')
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','APPL',ICON_APPL)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','ICTYPE',CMAQsimtype)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','DATE',start_date_MCIP)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/icon/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/icon/scripts/run_icon_daybyday_profile_WCAS.csh")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/ICON_profile.ok") == False:
                terminal_exec(terminal, "./run_icon_daybyday_profile_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/icon/scripts/iconcsh.ok") == True:
                        break
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/icon")) < 1: # 文件生成数量是否正常
                    print(T.Now() + ' - '+' 错误! ICON结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
            print(T.Now() + ' - ' +'    ICON运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/ICON_profile.ok")
        
            terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/emis") # 创建排放清单文件夹
            terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
            time.sleep(0.2) 
            print(T.Now() + ' - ' + '    是否运行MEGAN生成生物源: '+str(runMEGAN))
            if runMEGAN == True:
                print(T.Now() + ' - ' +'    开始运行PreMEGAN.....')
                GRIDDESC = open(f"{WCAS_dir}simulations/{gridname}/GRIDDESC")  
                lines = GRIDDESC.readlines()  # 读一次存入，读多次会错误
                rowcolgrid_pre = lines[5].split(' ')  # 读取namelist相应行的数据
                rowcolgrid = []
                for i in rowcolgrid_pre:
                    if i != '':
                        rowcolgrid.append(i)
                GRIDDESC.close()
                namelist_premegan = f90nml.read(f"{WCAS_dir}namelists_cshfiles/prepmegan4cmaq.inp.temp")
                namelist_premegan['control']['domains'] = regrid_dom
                namelist_premegan['control']['start_lai_mnth'] = 1 # 直接处理全年的pre文件， 
                namelist_premegan['control']['end_lai_mnth'] = 12
                namelist_premegan['control']['wrf_dir'] = WRF_dir+'run/WRFoutput'
                namelist_premegan['control']['megan_dir'] = premeganinput_dir
                namelist_premegan['control']['out_dir'] = premeganoutput_dir
                namelist_premegan['windowdefs']['ncolsin'] = int(rowcolgrid[5])
                namelist_premegan['windowdefs']['nrowsin'] = int(rowcolgrid[6])
                namelist_premegan.write(f"{premegan_dir}prepmegan4cmaq.inp", force=True)
                # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                terminal_exec(terminal, f"cd {premegan_dir}")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_prepmegan4cmaq_WCAS.csh {premegan_dir}run_prepmegan4cmaq_WCAS.csh")
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_profile.ok") == False:
                    terminal_exec(terminal, "./run_prepmegan4cmaq_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    # if check_process_status('prepmegan4cmaq_ef.x') == False and check_process_status(
                    #         'prepmegan4cmaq_lai.x') == False and check_process_status('prepmegan4cmaq_pft.x') == False:
                    if os.path.exists(f"{premegan_dir}premegancsh.ok") == True:
                        break
                # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    PreMEGAN运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_profile.ok")
        
                print(T.Now() + ' - ' +'    开始运行MEGAN.....')
                terminal_exec(terminal, f"chmod u+x {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh") # 将megan的work脚本移动
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh {MEGAN_dir}work/")
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','start_date',emis_start_date)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','end_date',emis_end_date)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNHOME',MEGAN_dir)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNINP',premeganoutput_dir)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','GDNAM3D',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','dom',regrid_dom)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','CAMQdata_dir',CMAQdata_dir)
                terminal_exec(terminal, f"export pys_megan_start_date={emis_start_date}")
                terminal_exec(terminal, f"export pys_megan_end_date={emis_end_date}")
                terminal_exec(terminal, f"cd {MEGAN_dir}")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_megan_WCAS.csh {MEGAN_dir}run_megan_WCAS.csh")
                # # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_profile.ok") == False:
                    terminal_exec(terminal, "./run_megan_WCAS.csh")
                    time.sleep(1.5)
                    while (True):
                        time.sleep(1)
                        # if check_process_status('emproc') == False and check_process_status(
                        #         'txt2ioapi') == False and check_process_status(
                        #     'met2mgn') == False and check_process_status('mgn2mech') == False:
                        if os.path.exists(f"{MEGAN_dir}megancsh.ok") == True:
                            break
                    terminal_exec(terminal, f"mv {MEGAN_dir}Output/MEGAN210.{MCIP_GridName}.*.ncf {CMAQdata_dir}{MCIP_GridName}/emis/")
                    # # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                    time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                    if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/emis")) < run_days_MCIP: # 文件生成数量是否正常
                        print(T.Now() + ' - '+' 错误! MEGAN生物源清单未能正常输出，请检查参数输入是否正确')
                        sys.exit() 
                print(T.Now() + ' - ' +'    MEGAN运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_profile.ok")
        
            if MEIAT_Linux == False: # Win下运行的带有arcgispro进行空间分配的MEIAT，还是直接在Linux调用仅平均分配的MEIAT
                print(T.Now() + ' - ' +'    开始主机调用MEIAT生成MEIC人为源排放清单.....')
                terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_dir}input/") # 传递GRIDDESC
                time.sleep(0.2)
                namelist_MEIAT = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.temp")
                namelist_MEIAT['global']['griddesc_name'] = [MCIP_GridName]
                namelist_MEIAT['global']['start_date'] = [emis_start_date]
                namelist_MEIAT['global']['end_date'] = [emis_end_date]
                namelist_MEIAT['global']['cores'] = [cores_win]
                namelist_MEIAT.write(f"{MEIAT_dir}namelist.input", force=True)
                terminal_exec(terminal, f"cd {MEIAT_dir}") # 进入MEIAT并执行排放清单计算
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_profile.ok") == False:
                    if os.path.exists(f"{MEIAT_dir}output/factor") == True: # 初始化MEIAT
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/factor") 
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/zoning_statistics")
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/source")
                    terminal_exec(terminal, f"cmd.exe /c {MEIAT_python_dir} ./coarse_emission_2_fine_emission.py") # 不知为何无法继续调用cmd.exe /c python ./Create-CMAQ-Emission-File.py，已经统一到coarse_emission_2_fine_emission中运行
                    time.sleep(5)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{MEIAT_dir}coarse_emission_2_fine_emission.OK") == True and os.path.exists(f"{MEIAT_dir}Create-CMAQ-Emission-File.OK") == True: # 等待运行
                            break
                    time.sleep(2)
                    terminal_exec(terminal, f"rm -f coarse_emission_2_fine_emission.OK") # 删除flag文件
                    terminal_exec(terminal, f"rm -f Create-CMAQ-Emission-File.OK") # 删除flag文件
                    terminal_exec(terminal, f"mv {MEIAT_dir}output/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                    time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    人为源排放清单生成完成')
            else:
                print(T.Now() + ' - ' +'    开始调用MEIAT_Linux生成MEIC人为源排放清单.....')
                terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_Linux_dir}input/") # 传递GRIDDESC
                time.sleep(0.2)
                namelist_MEIAT_Linux = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.Linux.temp")
                namelist_MEIAT_Linux['global']['griddesc_name'] = [MCIP_GridName]
                namelist_MEIAT_Linux['global']['start_date'] = [emis_start_date]
                namelist_MEIAT_Linux['global']['end_date'] = [emis_end_date]
                namelist_MEIAT_Linux['global']['cores'] = [cores_win]
                namelist_MEIAT_Linux.write(f"{MEIAT_Linux_dir}namelist.input", force=True)
                terminal_exec(terminal, f"cd {MEIAT_Linux_dir}") # 进入MEIAT并执行排放清单计算
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_profile.ok") == False:
                    terminal_exec(terminal, f"python3 meicmix_f2c.py") 
                    time.sleep(5)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{MEIAT_Linux_dir}meicmix_f2c.OK") == True: # 等待运行
                            break
                    time.sleep(2)
                    terminal_exec(terminal, f"rm -f meicmix_f2c.OK") # 删除flag文件
                    terminal_exec(terminal, f"mv {MEIAT_Linux_dir}model_emission_{MCIP_GridName}/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                    time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    人为源排放清单生成完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_profile.ok")
        
            print(T.Now() + ' - ' +'    开始运行CCTM.....')
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','VRSN',f"v{CCTMversion}")
            CMAQ_exe = f"CCTM_v{CCTMversion}.exe"
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','APPL',ICON_APPL)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','START_DATE',cctm_start_date)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','END_DATE',cctm_end_date)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','cores_CMAQ_col',cores_CMAQ_col)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','cores_CMAQ_row',cores_CMAQ_row)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CONC_level',"ONE") # 直接模拟里层不用CONC
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','N_EMIS_GR',5)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','runMEGAN',"N")
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CTM_ISAM',"N")
            if runMEGAN == True: 
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','N_EMIS_GR',6)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','runMEGAN',"Y")
            if runISAM == True:  # 是否开启ISAM
                print(T.Now() + ' - ' +'    开始ISAM输入数据准备')
                mcipfiles = os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip/")
                for f in mcipfiles:
                    if f.split("_")[0] == "GRIDCRO2D":
                        GRIDCRO2D_f = f 
                    if f == "GRIDDESC":
                        GRIDDESC_f = f
                terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/ISAM") # 创建ISAM配置文件夹
                if if_makemask == True:
                    ISAM_REGIONS_create(
                        terminal = terminal,
                        GRIDCRO2D_dir=f"{CMAQdata_dir}{MCIP_GridName}/mcip/{GRIDCRO2D_f}",
                        GRIDDECfile_dir=f"{CMAQdata_dir}{MCIP_GridName}/mcip/{GRIDDESC_f}",
                        Regions_varnames=mask_varnames_l,
                        target_dir = f"{CMAQdata_dir}{MCIP_GridName}/ISAM/"
                    )
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','VRSN',f"v{CCTMversion}_ISAM")
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CTM_ISAM',"Y")
                CMAQ_exe = f"CCTM_v{CCTMversion}_ISAM.exe"
                terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/ISAM_CONTROL.txt {CMAQdata_dir}{MCIP_GridName}/ISAM") # 复制配置文件
                # terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/ISAM_REGION.nc {CMAQdata_dir}{MCIP_GridName}/ISAM") 
                terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/EmissCtrl_cb6r3_ae6_aq.nml {CMAQ_dir}CCTM/scripts/BLD_CCTM_v{CCTMversion}_ISAM_{compiler_str}/") 
                print(T.Now() + ' - ' +'    ISAM预处理过程完成')
            terminal_exec(terminal, f"cd {CMAQ_dir}/CCTM/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh {CMAQ_dir}CCTM/scripts/")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/CCTM_profile.ok") == False:
                terminal_exec(terminal, f"./run_cctm_daybyday_profile_WCAS.csh")
                time.sleep(5)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}CCTM/scripts/cctmcsh.ok") == True:
                        break
            print(T.Now() + ' - ' +'    CCTM运行完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/CCTM_profile.ok")

        
        # ===============================================================================================================================
        # regrid: 以外层作为边界场模拟，边界条件BCON通过D01的模拟结果生成，边界源的远距离传输能够被捕捉，模拟结果更好
        # ===============================================================================================================================
        if CMAQsimtype == 'regrid': 
            print(T.Now() + ' - ' + f'开始准备CMAQ的模拟 - 模拟方式 - {CMAQsimtype}')
        
            # ===============================================================================================================================
            # D01远距离传输场模拟，用于生成非清洁空气边界条件
            # ===============================================================================================================================
            print(T.Now() + ' - ' +'    开始运行MCIP - d01')
            MCIP_GridName = gridname + '_d01'
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','CMAQ_HOME',CMAQ_dir)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','start_time',start_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','end_time',end_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','domain',1) # d01
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','GridName',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','DataPath',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InMetDir',f"{WRF_dir}run/WRFoutput")
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InGeoDir',f"{WCAS_dir}simulations/{gridname}/WPSoutput")
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','WRF_LC_REF_LAT',ref_lat)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/mcip/scripts/") #
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/mcip/scripts/run_mcip_daybyday_profile_WCAS.csh")
            # ========================================MCIP======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MCIP_regrid_d01.ok") == False:
                terminal_exec(terminal, f"./run_mcip_daybyday_profile_WCAS.csh {compiler_str}")
                time.sleep(3)
                while (True):
                    time.sleep(3)
                    if os.path.exists(f"{CMAQ_dir}PREP/mcip/scripts/mcipcsh.ok") == True:
                        break 
            if os.path.exists(f"{CMAQdata_dir}{MCIP_GridName}/mcip") == True: # 防止找不到路径报错 
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip")) < 2+run_days_MCIP*9: # 文件生成数量是否正常(2个单独的和9类)
                    print(T.Now() + ' - '+' 错误! MCIP结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
            # ========================================MCIP======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    MCIP运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MCIP_regrid_d01.ok")

            print(T.Now() + ' - ' +'    开始运行BCON - d01')
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','BCTYPE','profile') # d01
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','start_time',start_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','end_time',end_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/bcon/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_bcon_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/bcon/scripts/run_bcon_daybyday_profile_WCAS.csh")
            # ========================================BCON======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/BCON_regrid_d01.ok") == False:
                terminal_exec(terminal, "./run_bcon_daybyday_profile_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/bcon/scripts/bconcsh.ok") == True:
                        break
            if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/bcon")) < run_days_MCIP*1: # 文件生成数量是否正常
                print(T.Now() + ' - '+' 错误! BCON结果未能正常输出，请检查参数输入是否正确')
                sys.exit() 
            # ========================================BCON======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    BCON运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/BCON_regrid_d01.ok")

            print(T.Now() + ' - ' +'    开始运行ICON - d01')
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','APPL',ICON_APPL)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','ICTYPE','profile') # d01
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh','DATE',start_date_MCIP)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/icon/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_icon_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/icon/scripts/run_icon_daybyday_profile_WCAS.csh")
            # ========================================ICON======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/ICON_regrid_d01.ok") == False:
                terminal_exec(terminal, "./run_icon_daybyday_profile_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/icon/scripts/iconcsh.ok") == True:
                        break
            if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/icon")) < 1: # 文件生成数量是否正常
                print(T.Now() + ' - '+' 错误! ICON结果未能正常输出，请检查参数输入是否正确')
                sys.exit()
            # ========================================ICON======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    ICON运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/ICON_regrid_d01.ok")

            terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/emis") # 创建排放清单文件夹
            terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
            time.sleep(0.2)
            print(T.Now() + ' - ' + '    是否运行MEGAN生成生物源: '+str(runMEGAN))
            if runMEGAN == True:
                print(T.Now() + ' - ' +'    开始运行PreMEGAN - d01')
                GRIDDESC = open(f"{WCAS_dir}simulations/{gridname}/GRIDDESC")  
                lines = GRIDDESC.readlines()  # 读一次存入，读多次会错误
                rowcolgrid_pre = lines[5].split(' ')  # 读取namelist相应行的数据
                rowcolgrid = []
                for i in rowcolgrid_pre:
                    if i != '':
                        rowcolgrid.append(i)
                GRIDDESC.close()
                namelist_premegan = f90nml.read(f"{WCAS_dir}namelists_cshfiles/prepmegan4cmaq.inp.temp")
                namelist_premegan['control']['domains'] = 1 #!!!!!!!!!!!!!!!!!!!!!!!!!d01!!!!!!!!!!!!!!!!!!
                namelist_premegan['control']['start_lai_mnth'] = 1 # 直接处理全年的pre文件， 
                namelist_premegan['control']['end_lai_mnth'] = 12
                namelist_premegan['control']['wrf_dir'] = WRF_dir+'run/WRFoutput'
                namelist_premegan['control']['megan_dir'] = premeganinput_dir
                namelist_premegan['control']['out_dir'] = premeganoutput_dir
                namelist_premegan['windowdefs']['ncolsin'] = int(rowcolgrid[5])
                namelist_premegan['windowdefs']['nrowsin'] = int(rowcolgrid[6])
                namelist_premegan.write(f"{premegan_dir}prepmegan4cmaq.inp", force=True)
                # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                terminal_exec(terminal, f"cd {premegan_dir}")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_prepmegan4cmaq_WCAS.csh {premegan_dir}run_prepmegan4cmaq_WCAS.csh")
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_regrid_d01.ok") == False:
                    terminal_exec(terminal, "./run_prepmegan4cmaq_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    # if check_process_status('prepmegan4cmaq_ef.x') == False and check_process_status(
                    #         'prepmegan4cmaq_lai.x') == False and check_process_status('prepmegan4cmaq_pft.x') == False:
                    if os.path.exists(f"{premegan_dir}premegancsh.ok") == True:
                        break
                # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    PreMEGAN运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_regrid_d01.ok")
        
                print(T.Now() + ' - ' +'    开始运行MEGAN - d01')
                terminal_exec(terminal, f"chmod u+x {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh") # 将megan的work脚本移动
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh {MEGAN_dir}work/")
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','start_date',emis_start_date)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','end_date',emis_end_date)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNHOME',MEGAN_dir)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNINP',premeganoutput_dir)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','GDNAM3D',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','dom',1)   #!!!!!!!!!!!!!!!!!!!!!!!!!d01!!!!!!!!!!!!!!!!!!
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','CAMQdata_dir',CMAQdata_dir)
                terminal_exec(terminal, f"export pys_megan_start_date={emis_start_date}")
                terminal_exec(terminal, f"export pys_megan_end_date={emis_end_date}")
                terminal_exec(terminal, f"cd {MEGAN_dir}")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_megan_WCAS.csh {MEGAN_dir}run_megan_WCAS.csh")
                # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_regrid_d01.ok") == False:
                    terminal_exec(terminal, "./run_megan_WCAS.csh")
                time.sleep(1.5)
                while (True):
                    time.sleep(1)
                    # if check_process_status('emproc') == False and check_process_status(
                    #         'txt2ioapi') == False and check_process_status(
                    #     'met2mgn') == False and check_process_status('mgn2mech') == False:
                    if os.path.exists(f"{MEGAN_dir}megancsh.ok") == True:
                        break
                terminal_exec(terminal, f"mv {MEGAN_dir}Output/MEGAN210.{MCIP_GridName}.*.ncf {CMAQdata_dir}{MCIP_GridName}/emis/")
                # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                time.sleep(4)
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/emis")) < run_days_MCIP: # 文件生成数量是否正常
                    print(T.Now() + ' - '+' 错误! MEGAN生物源清单未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
                print(T.Now() + ' - ' +'    MEGAN运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_regrid_d01.ok")

            if MEIAT_Linux == False: # Win下运行的带有arcgispro进行空间分配的MEIAT，还是直接在Linux调用仅平均分配的MEIAT
                print(T.Now() + ' - ' +'    开始主机调用MEIAT生成MEIC人为源排放清单 - d01')
                terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_dir}input/") # 传递GRIDDESC
                time.sleep(0.2)
                namelist_MEIAT = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.temp")
                namelist_MEIAT['global']['griddesc_name'] = [MCIP_GridName]
                namelist_MEIAT['global']['start_date'] = [emis_start_date]
                namelist_MEIAT['global']['end_date'] = [emis_end_date]
                namelist_MEIAT['global']['cores'] = [cores_win]
                namelist_MEIAT.write(f"{MEIAT_dir}namelist.input", force=True)
                terminal_exec(terminal, f"cd {MEIAT_dir}") # 进入MEIAT并执行排放清单计算
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d01.ok") == False:
                    if os.path.exists(f"{MEIAT_dir}output/factor") == True: # 初始化MEIAT
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/factor") 
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/zoning_statistics")
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/source")
                    terminal_exec(terminal, f"rm -f fine_emission_2_coarse_emission.OK") # 删除flag文件
                    terminal_exec(terminal, f"cmd.exe /c {MEIAT_python_dir} ./fine_emission_2_coarse_emission.py")  # 模拟外层MEIAT只需要运行这一个程序
                    time.sleep(5)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{MEIAT_dir}fine_emission_2_coarse_emission.OK") == True: # 等待运行
                            break
                    terminal_exec(terminal, f"rm -f fine_emission_2_coarse_emission.OK") # 删除flag文件
                    terminal_exec(terminal, f"mmv {MEIAT_dir}output/'*_CB06_*.nc' {MEIAT_dir}output/'CB06_#2_#1.nc'") #MEIAT命名规范统一
                    time.sleep(3) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                    terminal_exec(terminal, f"mv {MEIAT_dir}output/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                    time.sleep(4) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    人为源排放清单生成完成')
            else:
                print(T.Now() + ' - ' +'    开始调用MEIAT_Linux生成MEIC人为源排放清单.....')
                terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_Linux_dir}input/") # 传递GRIDDESC
                time.sleep(0.2)
                namelist_MEIAT_Linux = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.Linux.temp")
                namelist_MEIAT_Linux['global']['griddesc_name'] = [MCIP_GridName]
                namelist_MEIAT_Linux['global']['start_date'] = [emis_start_date]
                namelist_MEIAT_Linux['global']['end_date'] = [emis_end_date]
                namelist_MEIAT_Linux['global']['cores'] = [cores_win]
                namelist_MEIAT_Linux.write(f"{MEIAT_Linux_dir}namelist.input", force=True)
                terminal_exec(terminal, f"cd {MEIAT_Linux_dir}") # 进入MEIAT并执行排放清单计算
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d01.ok") == False:
                    terminal_exec(terminal, f"python3 meicmix_f2c.py") 
                    time.sleep(5)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{MEIAT_Linux_dir}meicmix_f2c.OK") == True: # 等待运行
                            break
                    time.sleep(2)
                    terminal_exec(terminal, f"rm -f meicmix_f2c.OK") # 删除flag文件
                    terminal_exec(terminal, f"mv {MEIAT_Linux_dir}model_emission_{MCIP_GridName}/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                    time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    人为源排放清单生成完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d01.ok")

            print(T.Now() + ' - ' +'    开始运行CCTM - d01')
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','VRSN',F"v{CCTMversion}")
            CMAQ_exe = F"CCTM_v{CCTMversion}.exe"
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','APPL',ICON_APPL)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','START_DATE',cctm_start_date)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','END_DATE',cctm_end_date)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','cores_CMAQ_col',cores_CMAQ_col)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','cores_CMAQ_row',cores_CMAQ_row)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CONC_level',"ALL") # regrid 需要输出所有层
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','N_EMIS_GR',5)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','runMEGAN',"N")
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','CTM_ISAM',"N")
            if runMEGAN == True: 
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','N_EMIS_GR',6)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh','runMEGAN',"Y")
            # D01默认关闭ISAM
            terminal_exec(terminal, f"cd {CMAQ_dir}/CCTM/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_cctm_daybyday_profile_WCAS.csh {CMAQ_dir}CCTM/scripts/")
            # ========================================CMAQ======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/CCTM_regrid_d01.ok") == False:
                terminal_exec(terminal, f"./run_cctm_daybyday_profile_WCAS.csh")
                time.sleep(10)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}CCTM/scripts/cctmcsh.ok") == True:
                        break
            # ========================================CMAQ======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    CCTM-d01运行完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/CCTM_regrid_d01.ok")

            # ===============================================================================================================================
            # D02环境场模拟并输出结果，当regrid_D02andD03 == .true.时候进行
            # ===============================================================================================================================
            if regrid_D02andD03 == True:
                regrid_dom = 2
                print(T.Now() + ' - ' +'    开始运行MCIP - d0'+str(regrid_dom))
                MCIP_GridName = gridname + f'_d0{str(regrid_dom)}'
                MCIP_GridName_d01 = gridname + '_d01'
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','CMAQ_HOME',CMAQ_dir)
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','start_time',start_date_MCIP)
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','end_time',end_date_MCIP)
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','domain',regrid_dom) # 进行regrid的层，不一定是第3，也可以是2
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','GridName',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','DataPath',CMAQdata_dir)
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InMetDir',f"{WRF_dir}run/WRFoutput")
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InGeoDir',f"{WCAS_dir}simulations/{gridname}/WPSoutput")
                modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','WRF_LC_REF_LAT',ref_lat)
                terminal_exec(terminal, f"cd {CMAQ_dir}PREP/mcip/scripts/") #
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/mcip/scripts/run_mcip_daybyday_profile_WCAS.csh")
                # ========================================MCIP======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MCIP_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, f"./run_mcip_daybyday_profile_WCAS.csh {compiler_str}")
                    time.sleep(3)
                    while (True):
                        time.sleep(3)
                        if os.path.exists(f"{CMAQ_dir}PREP/mcip/scripts/mcipcsh.ok") == True:
                            break
                if os.path.exists(f"{CMAQdata_dir}{MCIP_GridName}/mcip") == True:
                    if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip")) < 2+run_days_MCIP*9: # 文件生成数量是否正常(2个单独的和9类)
                        print(T.Now() + ' - '+' 错误! MCIP结果未能正常输出，请检查参数输入是否正确')
                        sys.exit() 
                # ========================================MCIP======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    MCIP运行完成！')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MCIP_regrid_d0{regrid_dom}.ok")

                print(T.Now() + ' - ' +'    开始运行BCON - d0'+str(regrid_dom))
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','BCTYPE',CMAQsimtype) # d01
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','start_time',start_date_MCIP)
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','end_time',end_date_MCIP)
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','GRID_NAME',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','d01_GRID_NAME',MCIP_GridName_d01)
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
                modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','APPL',ICON_APPL)
                terminal_exec(terminal, f"cd {CMAQ_dir}PREP/bcon/scripts")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh {CMAQ_dir}PREP/bcon/scripts/run_bcon_daybyday_regrid_WCAS.csh")
                # ========================================BCON======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/BCON_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, "./run_bcon_daybyday_regrid_WCAS.csh")
                    time.sleep(2)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{CMAQ_dir}PREP/bcon/scripts/bconcsh.ok") == True:
                            break
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/bcon")) < run_days_MCIP*1: # 文件生成数量是否正常
                    print(T.Now() + ' - '+' 错误! BCON结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
                # ========================================BCON======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    BCON运行完成！')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/BCON_regrid_d0{regrid_dom}.ok")

                print(T.Now() + ' - ' +'    开始运行ICON - d0'+str(regrid_dom))
                modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','APPL',ICON_APPL)
                modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','ICTYPE',CMAQsimtype) # 
                modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','GRID_NAME',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','d01_GRID_NAME',MCIP_GridName_d01)
                modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
                modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','DATE',start_date_MCIP)
                terminal_exec(terminal, f"cd {CMAQ_dir}PREP/icon/scripts")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh {CMAQ_dir}PREP/icon/scripts/run_icon_daybyday_regrid_WCAS.csh")
                # ========================================ICON======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/ICON_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, "./run_icon_daybyday_regrid_WCAS.csh")
                    time.sleep(1)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{CMAQ_dir}PREP/icon/scripts/iconcsh.ok") == True:
                            break
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/icon")) < 1: # 文件生成数量是否正常
                    print(T.Now() + ' - '+' 错误! ICON结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
                # ========================================ICON======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    ICON运行完成！')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/ICON_regrid_d0{regrid_dom}.ok")

                terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/emis") # 创建排放清单文件夹
                terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
                time.sleep(0.2)
                print(T.Now() + ' - ' + '    是否运行MEGAN生成生物源: '+str(runMEGAN))
                if runMEGAN == True:
                    print(T.Now() + ' - ' +'    开始运行PreMEGAN - d0'+str(regrid_dom))
                    GRIDDESC = open(f"{WCAS_dir}simulations/{gridname}/GRIDDESC")  
                    lines = GRIDDESC.readlines()  # 读一次存入，读多次会错误
                    rowcolgrid_pre = lines[5].split(' ')  # 读取namelist相应行的数据
                    rowcolgrid = []
                    for i in rowcolgrid_pre:
                        if i != '':
                            rowcolgrid.append(i)
                    GRIDDESC.close()
                    namelist_premegan = f90nml.read("namelists_cshfiles/prepmegan4cmaq.inp.temp")
                    namelist_premegan['control']['domains'] = regrid_dom #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    namelist_premegan['control']['start_lai_mnth'] = 1 # 直接处理全年的pre文件， 
                    namelist_premegan['control']['end_lai_mnth'] = 12
                    namelist_premegan['control']['wrf_dir'] = WRF_dir+'run/WRFoutput'
                    namelist_premegan['control']['megan_dir'] = premeganinput_dir
                    namelist_premegan['control']['out_dir'] = premeganoutput_dir
                    namelist_premegan['windowdefs']['ncolsin'] = int(rowcolgrid[5])
                    namelist_premegan['windowdefs']['nrowsin'] = int(rowcolgrid[6])
                    namelist_premegan.write(f"{premegan_dir}prepmegan4cmaq.inp", force=True)
                    # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                    terminal_exec(terminal, f"cd {premegan_dir}")
                    terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_prepmegan4cmaq_WCAS.csh {premegan_dir}run_prepmegan4cmaq_WCAS.csh")
                    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_regrid_d0{regrid_dom}.ok") == False:
                        terminal_exec(terminal, "./run_prepmegan4cmaq_WCAS.csh")
                    while (True):
                        time.sleep(1)
                        # if check_process_status('prepmegan4cmaq_ef.x') == False and check_process_status(
                        #         'prepmegan4cmaq_lai.x') == False and check_process_status('prepmegan4cmaq_pft.x') == False:
                        if os.path.exists(f"{premegan_dir}premegancsh.ok") == True:
                            break
                    # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                    print(T.Now() + ' - ' +'    PreMEGAN运行完成')
                    terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_regrid_d0{regrid_dom}.ok")
            
                    print(T.Now() + ' - ' +'    开始运行MEGAN - d0'+str(regrid_dom))
                    terminal_exec(terminal, f"chmod u+x {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh") # 将megan的work脚本移动
                    terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh {MEGAN_dir}work/")
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','start_date',emis_start_date)
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','end_date',emis_end_date)
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNHOME',MEGAN_dir)
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNINP',premeganoutput_dir)
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','GDNAM3D',MCIP_GridName)
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','dom',regrid_dom)   #!!!!!!!!!!!!!!!!!!!!!!!!!d01!!!!!!!!!!!!!!!!!!
                    modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','CAMQdata_dir',CMAQdata_dir)
                    terminal_exec(terminal, f"cd {MEGAN_dir}")
                    terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_megan_WCAS.csh {MEGAN_dir}run_megan_WCAS.csh")
                    # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_regrid_d0{regrid_dom}.ok") == False:
                        terminal_exec(terminal, "./run_megan_WCAS.csh")
                    while (True):
                        time.sleep(1)
                        # if check_process_status('emproc') == False and check_process_status(
                        #         'txt2ioapi') == False and check_process_status(
                        #     'met2mgn') == False and check_process_status('mgn2mech') == False:
                        if os.path.exists(f"{MEGAN_dir}megancsh.ok") == True:
                            break
                    terminal_exec(terminal, f"mv {MEGAN_dir}Output/MEGAN210.{MCIP_GridName}.*.ncf {CMAQdata_dir}{MCIP_GridName}/emis/")
                    # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                    time.sleep(4)
                    if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/emis")) < run_days_MCIP: # 文件生成数量是否正常
                        print(T.Now() + ' - '+' 错误! MEGAN生物源清单未能正常输出，请检查参数输入是否正确')
                        sys.exit() 
                    print(T.Now() + ' - ' +'    MEGAN运行完成')
                    terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_regrid_d0{regrid_dom}.ok")
                
                if MEIAT_Linux == False: # Win下运行的带有arcgispro进行空间分配的MEIAT，还是直接在Linux调用仅平均分配的MEIAT
                    print(T.Now() + ' - ' +'    开始主机调用MEIAT生成MEIC人为源排放清单.....')
                    terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_dir}input/") # 传递GRIDDESC
                    time.sleep(0.2)
                    namelist_MEIAT = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.temp")
                    namelist_MEIAT['global']['griddesc_name'] = [MCIP_GridName]
                    namelist_MEIAT['global']['start_date'] = [emis_start_date]
                    namelist_MEIAT['global']['end_date'] = [emis_end_date]
                    namelist_MEIAT['global']['cores'] = [cores_win]
                    namelist_MEIAT.write(f"{MEIAT_dir}namelist.input", force=True)
                    terminal_exec(terminal, f"cd {MEIAT_dir}") # 进入MEIAT并执行排放清单计算
                    # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d0{regrid_dom}.ok") == False:
                        if os.path.exists(f"{MEIAT_dir}output/factor") == True: # 初始化MEIAT
                            terminal_exec(terminal, f"rm -r {MEIAT_dir}output/factor") 
                            terminal_exec(terminal, f"rm -r {MEIAT_dir}output/zoning_statistics")
                            terminal_exec(terminal, f"rm -r {MEIAT_dir}output/source")
                        terminal_exec(terminal, f"rm -f coarse_emission_2_fine_emission.OK") # 删除flag文件
                        terminal_exec(terminal, f"rm -f Create-CMAQ-Emission-File.OK") # 删除flag文件
                        terminal_exec(terminal, f"cmd.exe /c {MEIAT_python_dir} ./coarse_emission_2_fine_emission.py") # 不知为何无法继续调用cmd.exe /c python ./Create-CMAQ-Emission-File.py，已经统一到coarse_emission_2_fine_emission中运行
                        time.sleep(5)
                        while (True):
                            time.sleep(1)
                            if os.path.exists(f"{MEIAT_dir}coarse_emission_2_fine_emission.OK") == True and os.path.exists(f"{MEIAT_dir}Create-CMAQ-Emission-File.OK") == True: # 等待运行
                                break
                        time.sleep(2)
                        terminal_exec(terminal, f"rm -f coarse_emission_2_fine_emission.OK") # 删除flag文件
                        terminal_exec(terminal, f"rm -f Create-CMAQ-Emission-File.OK") # 删除flag文件
                        terminal_exec(terminal, f"mv {MEIAT_dir}output/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                        time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                    # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                    print(T.Now() + ' - ' +'    人为源排放清单生成完成')
                else:
                    print(T.Now() + ' - ' +'    开始调用MEIAT_Linux生成MEIC人为源排放清单.....')
                    terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_Linux_dir}input/") # 传递GRIDDESC
                    time.sleep(0.2)
                    namelist_MEIAT_Linux = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.Linux.temp")
                    namelist_MEIAT_Linux['global']['griddesc_name'] = [MCIP_GridName]
                    namelist_MEIAT_Linux['global']['start_date'] = [emis_start_date]
                    namelist_MEIAT_Linux['global']['end_date'] = [emis_end_date]
                    namelist_MEIAT_Linux['global']['cores'] = [cores_win]
                    namelist_MEIAT_Linux.write(f"{MEIAT_Linux_dir}namelist.input", force=True)
                    terminal_exec(terminal, f"cd {MEIAT_Linux_dir}") # 进入MEIAT并执行排放清单计算
                    # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d0{regrid_dom}.ok") == False:
                        terminal_exec(terminal, f"python3 meicmix_f2c.py") 
                        time.sleep(5)
                        while (True):
                            time.sleep(1)
                            if os.path.exists(f"{MEIAT_Linux_dir}meicmix_f2c.OK") == True: # 等待运行
                                break
                        time.sleep(2)
                        terminal_exec(terminal, f"rm -f meicmix_f2c.OK") # 删除flag文件
                        terminal_exec(terminal, f"mv {MEIAT_Linux_dir}model_emission_{MCIP_GridName}/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                        time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                    # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                    print(T.Now() + ' - ' +'    人为源排放清单生成完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d0{regrid_dom}.ok")

                print(T.Now() + ' - ' +'    开始运行CCTM - d0'+str(regrid_dom))
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','VRSN',F"v{CCTMversion}")
                CMAQ_exe = F"CCTM_v{CCTMversion}.exe"
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','APPL',ICON_APPL)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','GRID_NAME',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','START_DATE',cctm_start_date)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','END_DATE',cctm_end_date)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','cores_CMAQ_col',cores_CMAQ_col)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','cores_CMAQ_row',cores_CMAQ_row)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CONC_level',"ONE") # 直接模拟里层不用CONC
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','N_EMIS_GR',5)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','runMEGAN',"N")
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CTM_ISAM',"N")
                if runMEGAN == True: 
                    modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','N_EMIS_GR',6)
                    modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','runMEGAN',"Y")
                if runISAM == True:  # 是否开启ISAM
                    mcipfiles = os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip/")
                    for f in mcipfiles:
                        if f.split("_")[0] == "GRIDCRO2D":
                            GRIDCRO2D_f = f 
                        if f == "GRIDDESC":
                            GRIDDESC_f = f
                    if if_makemask == True:
                        ISAM_REGIONS_create(
                            terminal = terminal,
                            GRIDCRO2D_dir=f"{CMAQdata_dir}{MCIP_GridName}/mcip/{GRIDCRO2D_f}",
                            GRIDDECfile_dir=f"{CMAQdata_dir}{MCIP_GridName}/mcip/{GRIDDESC_f}",
                            Regions_varnames=mask_varnames,
                        )
                    modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','VRSN',F"{CCTMversion}_ISAM")
                    modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CTM_ISAM',"Y")
                    CMAQ_exe = f"CCTM_v{CCTMversion}_ISAM.exe"
                    terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/ISAM") # 创建ISAM配置文件夹
                    terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/ISAM_CONTROL.txt {CMAQdata_dir}{MCIP_GridName}/ISAM") # 复制配置文件
                    # terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/ISAM_REGION.nc {CMAQdata_dir}{MCIP_GridName}/ISAM") 
                    terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/EmissCtrl_cb6r3_ae6_aq.nml {CMAQ_dir}CCTM/scripts/BLD_CCTM_v{CCTMversion}_ISAM_{compiler_str}/") 
                    time.sleep(3)
                terminal_exec(terminal, f"cd {CMAQ_dir}/CCTM/scripts")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh {CMAQ_dir}CCTM/scripts/")
                # ========================================CMAQ======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/CCTM_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, f"./run_cctm_daybyday_regrid_WCAS.csh")
                    time.sleep(15)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{CMAQ_dir}CCTM/scripts/cctmcsh.ok") == True:
                            break
                # ========================================CMAQ======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    CCTM运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/CCTM_regrid_d0{regrid_dom}.ok")
                regrid_dom = 3  # 继续进行下一层第三层的模拟

            print(T.Now() + ' - ' +'    开始运行MCIP - d0'+str(regrid_dom))
            MCIP_GridName = gridname + f'_d0{str(regrid_dom)}'
            MCIP_GridName_d01 = gridname + '_d01'
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','CMAQ_HOME',CMAQ_dir)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','start_time',start_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','end_time',end_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','domain',regrid_dom) # 进行regrid的层，不一定是第3，也可以是2
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','GridName',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','DataPath',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InMetDir',f"{WRF_dir}run/WRFoutput")
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','InGeoDir',f"{WCAS_dir}simulations/{gridname}/WPSoutput")
            modify_csh_variable('namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh','WRF_LC_REF_LAT',ref_lat)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/mcip/scripts/") #
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_mcip_daybyday_profile_WCAS.csh {CMAQ_dir}PREP/mcip/scripts/run_mcip_daybyday_profile_WCAS.csh")
            # ========================================MCIP======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MCIP_regrid_d0{regrid_dom}.ok") == False:
                terminal_exec(terminal, f"./run_mcip_daybyday_profile_WCAS.csh {compiler_str}")
                time.sleep(3)
                while (True):
                    time.sleep(3)
                    if os.path.exists(f"{CMAQ_dir}PREP/mcip/scripts/mcipcsh.ok") == True:
                        break
            if os.path.exists(f"{CMAQdata_dir}{MCIP_GridName}/mcip") == True: #防止找不到路径报错
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip")) < 2+run_days_MCIP*9 : # 文件生成数量是否正常(2个单独的和9类)
                    print(T.Now() + ' - '+' 错误! MCIP结果未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
            # ========================================MCIP======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    MCIP运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MCIP_regrid_d0{regrid_dom}.ok")

            print(T.Now() + ' - ' +'    开始运行BCON - d0'+str(regrid_dom))
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','BCTYPE',CMAQsimtype) # d01
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','start_time',start_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','end_time',end_date_MCIP)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','d01_GRID_NAME',MCIP_GridName_d01)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh','APPL',ICON_APPL)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/bcon/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_bcon_daybyday_regrid_WCAS.csh {CMAQ_dir}PREP/bcon/scripts/run_bcon_daybyday_regrid_WCAS.csh")
            # ========================================BCON======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/BCON_regrid_d0{regrid_dom}.ok") == False:
                terminal_exec(terminal, "./run_bcon_daybyday_regrid_WCAS.csh")
                time.sleep(2)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/bcon/scripts/bconcsh.ok") == True:
                        break
            if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/bcon")) < run_days_MCIP*1: # 文件生成数量是否正常
                print(T.Now() + ' - '+' 错误! BCON结果未能正常输出，请检查参数输入是否正确')
                sys.exit() 
            # ========================================BCON======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    BCON运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/BCON_regrid_d0{regrid_dom}.ok")

            print(T.Now() + ' - ' +'    开始运行ICON - d0'+str(regrid_dom))
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','APPL',ICON_APPL)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','ICTYPE',CMAQsimtype) # 
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','d01_GRID_NAME',MCIP_GridName_d01)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh','DATE',start_date_MCIP)
            terminal_exec(terminal, f"cd {CMAQ_dir}PREP/icon/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_icon_daybyday_regrid_WCAS.csh {CMAQ_dir}PREP/icon/scripts/run_icon_daybyday_regrid_WCAS.csh")
            # ========================================ICON======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/ICON_regrid_d0{regrid_dom}.ok") == False:
                terminal_exec(terminal, "./run_icon_daybyday_regrid_WCAS.csh")
                time.sleep(1)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}PREP/icon/scripts/iconcsh.ok") == True:
                        break
            if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/icon")) < 1: # 文件生成数量是否正常
                print(T.Now() + ' - '+' 错误! ICON结果未能正常输出，请检查参数输入是否正确')
                sys.exit() 
            # ========================================ICON======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    ICON运行完成！')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/ICON_regrid_d0{regrid_dom}.ok")

            terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/emis") # 创建排放清单文件夹
            terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
            time.sleep(0.2)
            print(T.Now() + ' - ' + '    是否运行MEGAN生成生物源: '+str(runMEGAN))
            if runMEGAN == True:
                print(T.Now() + ' - ' +'    开始运行PreMEGAN - d0'+str(regrid_dom))
                GRIDDESC = open(f"{WCAS_dir}simulations/{gridname}/GRIDDESC")  
                lines = GRIDDESC.readlines()  # 读一次存入，读多次会错误
                rowcolgrid_pre = lines[5].split(' ')  # 读取namelist相应行的数据
                rowcolgrid = []
                for i in rowcolgrid_pre:
                    if i != '':
                        rowcolgrid.append(i)
                GRIDDESC.close()
                namelist_premegan = f90nml.read("namelists_cshfiles/prepmegan4cmaq.inp.temp")
                namelist_premegan['control']['domains'] = regrid_dom #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                namelist_premegan['control']['start_lai_mnth'] = 1 # 直接处理全年的pre文件， 
                namelist_premegan['control']['end_lai_mnth'] = 12
                namelist_premegan['control']['wrf_dir'] = WRF_dir+'run/WRFoutput'
                namelist_premegan['control']['megan_dir'] = premeganinput_dir
                namelist_premegan['control']['out_dir'] = premeganoutput_dir
                namelist_premegan['windowdefs']['ncolsin'] = int(rowcolgrid[5])
                namelist_premegan['windowdefs']['nrowsin'] = int(rowcolgrid[6])
                namelist_premegan.write(f"{premegan_dir}prepmegan4cmaq.inp", force=True)
                # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                terminal_exec(terminal, f"cd {premegan_dir}")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_prepmegan4cmaq_WCAS.csh {premegan_dir}run_prepmegan4cmaq_WCAS.csh")
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, "./run_prepmegan4cmaq_WCAS.csh")
                while (True):
                    time.sleep(1)
                    # if check_process_status('prepmegan4cmaq_ef.x') == False and check_process_status(
                    #         'prepmegan4cmaq_lai.x') == False and check_process_status('prepmegan4cmaq_pft.x') == False:
                    if os.path.exists(f"{premegan_dir}premegancsh.ok") == True:
                        break
                # ========================================preMEGAN======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    PreMEGAN运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/PreMEGAN_regrid_d0{regrid_dom}.ok")
        
                print(T.Now() + ' - ' +'    开始运行MEGAN - d0'+str(regrid_dom))
                terminal_exec(terminal, f"chmod u+x {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh") # 将megan的work脚本移动
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/meganwork_scripts/*.csh {MEGAN_dir}work/")
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','start_date',emis_start_date)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','end_date',emis_end_date)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNHOME',MEGAN_dir)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','MGNINP',premeganoutput_dir)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','GDNAM3D',MCIP_GridName)
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','dom',regrid_dom)   #!!!!!!!!!!!!!!!!!!!!!!!!!d01!!!!!!!!!!!!!!!!!!
                modify_csh_variable('namelists_cshfiles/run_megan_WCAS.csh','CAMQdata_dir',CMAQdata_dir)
                terminal_exec(terminal, f"cd {MEGAN_dir}")
                terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_megan_WCAS.csh {MEGAN_dir}run_megan_WCAS.csh")
                # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, "./run_megan_WCAS.csh")
                while (True):
                    time.sleep(1)
                    # if check_process_status('emproc') == False and check_process_status(
                    #         'txt2ioapi') == False and check_process_status(
                    #     'met2mgn') == False and check_process_status('mgn2mech') == False:
                    if os.path.exists(f"{MEGAN_dir}megancsh.ok") == True:
                        break
                terminal_exec(terminal, f"mv {MEGAN_dir}Output/MEGAN210.{MCIP_GridName}.*.ncf {CMAQdata_dir}{MCIP_GridName}/emis/")
                # ========================================MEGAN======================================== 注释此部分可跳过,用于调试
                time.sleep(4)
                if len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/emis")) < run_days_MCIP: # 文件生成数量是否正常
                    print(T.Now() + ' - '+' 错误! MEGAN生物源清单未能正常输出，请检查参数输入是否正确')
                    sys.exit() 
                print(T.Now() + ' - ' +'    MEGAN运行完成')
                terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEGAN_regrid_d0{regrid_dom}.ok")
            
            if MEIAT_Linux == False: # Win下运行的带有arcgispro进行空间分配的MEIAT，还是直接在Linux调用仅平均分配的MEIAT
                print(T.Now() + ' - ' +'    开始主机调用MEIAT生成MEIC人为源排放清单.....')
                terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_dir}input/") # 传递GRIDDESC
                time.sleep(0.2)
                namelist_MEIAT = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.temp")
                namelist_MEIAT['global']['griddesc_name'] = [MCIP_GridName]
                namelist_MEIAT['global']['start_date'] = [emis_start_date]
                namelist_MEIAT['global']['end_date'] = [emis_end_date]
                namelist_MEIAT['global']['cores'] = [cores_win]
                namelist_MEIAT.write(f"{MEIAT_dir}namelist.input", force=True)
                terminal_exec(terminal, f"cd {MEIAT_dir}") # 进入MEIAT并执行排放清单计算
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d0{regrid_dom}.ok") == False:
                    if os.path.exists(f"{MEIAT_dir}output/factor") == True: # 初始化MEIAT
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/factor") 
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/zoning_statistics")
                        terminal_exec(terminal, f"rm -r {MEIAT_dir}output/source")
                    terminal_exec(terminal, f"rm -f coarse_emission_2_fine_emission.OK") # 删除flag文件
                    terminal_exec(terminal, f"rm -f Create-CMAQ-Emission-File.OK") # 删除flag文件
                    terminal_exec(terminal, f"cmd.exe /c {MEIAT_python_dir} ./coarse_emission_2_fine_emission.py") # 不知为何无法继续调用cmd.exe /c python ./Create-CMAQ-Emission-File.py，已经统一到coarse_emission_2_fine_emission中运行
                    time.sleep(5)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{MEIAT_dir}coarse_emission_2_fine_emission.OK") == True and os.path.exists(f"{MEIAT_dir}Create-CMAQ-Emission-File.OK") == True: # 等待运行
                            break
                    time.sleep(2)
                    terminal_exec(terminal, f"rm -f coarse_emission_2_fine_emission.OK") # 删除flag文件
                    terminal_exec(terminal, f"rm -f Create-CMAQ-Emission-File.OK") # 删除flag文件
                    terminal_exec(terminal, f"mv {MEIAT_dir}output/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                    time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    人为源排放清单生成完成')
            else:
                print(T.Now() + ' - ' +'    开始调用MEIAT_Linux生成MEIC人为源排放清单.....')
                terminal_exec(terminal, f"cp {WCAS_dir}simulations/{gridname}/GRIDDESC {MEIAT_Linux_dir}input/") # 传递GRIDDESC
                time.sleep(0.2)
                namelist_MEIAT_Linux = f90nml.read(f"{WCAS_dir}namelists_cshfiles/namelist.MEIAT.Linux.temp")
                namelist_MEIAT_Linux['global']['griddesc_name'] = [MCIP_GridName]
                namelist_MEIAT_Linux['global']['start_date'] = [emis_start_date]
                namelist_MEIAT_Linux['global']['end_date'] = [emis_end_date]
                namelist_MEIAT_Linux['global']['cores'] = [cores_win]
                namelist_MEIAT_Linux.write(f"{MEIAT_Linux_dir}namelist.input", force=True)
                terminal_exec(terminal, f"cd {MEIAT_Linux_dir}") # 进入MEIAT并执行排放清单计算
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d0{regrid_dom}.ok") == False:
                    terminal_exec(terminal, f"python3 meicmix_f2c.py") 
                    time.sleep(5)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{MEIAT_Linux_dir}meicmix_f2c.OK") == True: # 等待运行
                            break
                    time.sleep(2)
                    terminal_exec(terminal, f"rm -f meicmix_f2c.OK") # 删除flag文件
                    terminal_exec(terminal, f"mv {MEIAT_Linux_dir}model_emission_{MCIP_GridName}/*_*_{MCIP_GridName}_*.nc {CMAQdata_dir}{MCIP_GridName}/emis/") # 移动生成的人为源排放清单
                    time.sleep(5) # 移动，复制等操作后需要等待时间，否则会在未完成复制移动后进行检测
                # ========================================MEIAT======================================== 注释此部分可跳过,用于调试
                print(T.Now() + ' - ' +'    人为源排放清单生成完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/MEIAT_regrid_d0{regrid_dom}.ok")

            print(T.Now() + ' - ' +'    开始运行CCTM - d0'+str(regrid_dom))
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','VRSN',F"v{CCTMversion}")
            CMAQ_exe = F"CCTM_v{CCTMversion}.exe"
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','APPL',ICON_APPL)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','GRID_NAME',MCIP_GridName)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CMAQdata_dir',CMAQdata_dir)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','START_DATE',cctm_start_date)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','END_DATE',cctm_end_date)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','cores_CMAQ_col',cores_CMAQ_col)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','cores_CMAQ_row',cores_CMAQ_row)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CONC_level',"ONE") # 直接模拟里层不用CONC
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','N_EMIS_GR',5)
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','runMEGAN',"N")
            modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CTM_ISAM',"N")
            if runMEGAN == True: 
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','N_EMIS_GR',6)
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','runMEGAN',"Y")
            if runISAM == True:  # 是否开启ISAM
                mcipfiles = os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip/")
                for f in mcipfiles:
                    if f.split("_")[0] == "GRIDCRO2D":
                        GRIDCRO2D_f = f 
                    if f == "GRIDDESC":
                        GRIDDESC_f = f
                if if_makemask == True:
                    ISAM_REGIONS_create(
                        terminal = terminal,
                        GRIDCRO2D_dir=f"{CMAQdata_dir}{MCIP_GridName}/mcip/{GRIDCRO2D_f}",
                        GRIDDECfile_dir=f"{CMAQdata_dir}{MCIP_GridName}/mcip/{GRIDDESC_f}",
                        Regions_varnames=mask_varnames,
                    )
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','VRSN',F"{CCTMversion}_ISAM")
                modify_csh_variable('namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh','CTM_ISAM',"Y")
                CMAQ_exe = f"CCTM_v{CCTMversion}_ISAM.exe"
                terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/ISAM") # 创建ISAM配置文件夹
                terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/ISAM_CONTROL.txt {CMAQdata_dir}{MCIP_GridName}/ISAM") # 复制配置文件
                # terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/ISAM_REGION.nc {CMAQdata_dir}{MCIP_GridName}/ISAM") 
                terminal_exec(terminal, f"cp {WCAS_dir}input/ISAM/EmissCtrl_cb6r3_ae6_aq.nml {CMAQ_dir}CCTM/scripts/BLD_CCTM_v{CCTMversion}_ISAM_{compiler_str}/") 
                time.sleep(3)
            terminal_exec(terminal, f"cd {CMAQ_dir}/CCTM/scripts")
            terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_cctm_daybyday_regrid_WCAS.csh {CMAQ_dir}CCTM/scripts/")
            # ========================================CMAQ======================================== 注释此部分可跳过,用于调试
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/CCTM_regrid_d0{regrid_dom}.ok") == False:
                terminal_exec(terminal, f"./run_cctm_daybyday_regrid_WCAS.csh")
                time.sleep(5)
                while (True):
                    time.sleep(1)
                    if os.path.exists(f"{CMAQ_dir}CCTM/scripts/cctmcsh.ok") == True:
                        break
            # ========================================CMAQ======================================== 注释此部分可跳过,用于调试
            print(T.Now() + ' - ' +'    CCTM运行完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/CCTM_regrid_d0{regrid_dom}.ok")
    
        print(T.Now() + ' - ' +f'WRF-CMAQ运行完成 运行方式: {CMAQsimtype} 运行耗时: {time.time() - WCAS_start_time}')
    else:
        print(T.Now() + ' - ' +f'WRF 运行完成  运行耗时: {time.time() - WCAS_start_time}')

    

    print(T.Now() + ' - ' + f'开始进行WRF-CMAQ后处理 ')
    if CMAQsimtype == 'nan' and runCMAQ == True:
        MCIP_GridName = gridname + f'_d0{str(regrid_dom)}'
        MCIP_GridName_d01 = gridname + '_d01'
        terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
        time.sleep(0.5)

    if runCMAQ:
        CMAQdata_dir = f"{WCAS_dir}simulations/{gridname}/"
        if regrid_D02andD03 == False:
            print(T.Now() + ' - ' + f'CombineCMAQ的结果...')
            MCIP_GridName = gridname + f'_d0{str(regrid_dom)}'
            MCIP_GridName_d01 = gridname + '_d01'
            terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/Post_combine.ok") == False:
                if CMAQcombine == False:  # 自己编写的Combine脚本，PM10还不能输出
                    terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/cctmCombine") # 整理结果文件
                    terminal_exec(terminal, f"cd {CMAQdata_dir}{MCIP_GridName}/cctm/") 
                    terminal_exec(terminal, f"mkdir toCombine")
                    terminal_exec(terminal, f"mv CCTM_ACONC_* toCombine")
                    terminal_exec(terminal, f"mv CCTM_APMDIAG_* toCombine")
                    time.sleep(5)
                    O3PM25Combine_CMAQfiles( 
                        CCTM_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctm/toCombine/",
                        Combine_file_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctmCombine/{MCIP_GridName}_combine_PM2503.nc",
                        GRIDDECfile_dir=f"{WCAS_dir}simulations/{gridname}/GRIDDESC",
                        start_time_yesterday=[1111,1,1] # 只是在nc文件中显示，可以忽略
                    )# 合并CMAQ文件
                else:   # 调用官方的combine脚本
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','APPL',ICON_APPL)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','GRID_NAME',MCIP_GridName)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','INPDIR',f'{CMAQdata_dir}{MCIP_GridName}/')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','METDIR',f'{CMAQdata_dir}{MCIP_GridName}/mcip')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','CCTMOUTDIR',f'{CMAQdata_dir}{MCIP_GridName}/cctm')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','POSTDIR',f'{CMAQdata_dir}{MCIP_GridName}/POST')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','START_DATE',cctm_start_date)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','END_DATE',cctm_end_date)
                    terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_combine_WCAS.csh {CMAQ_dir}POST/combine/scripts/")
                    time.sleep(0.8)
                    terminal_exec(terminal, f"cd {CMAQ_dir}POST/combine/scripts/")
                    terminal_exec(terminal, f"./run_combine_WCAS.csh")
                    time.sleep(2)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{CMAQ_dir}POST/combine/scripts/combinecsh.ok") == True:
                            break
            print(T.Now() + ' - ' + f'CMAQ结果Combine完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/Post_combine.ok")
        else:
            #D02：
            print(T.Now() + ' - ' + f'CombineCMAQ的结果... - d02')
            regrid_dom = 2
            MCIP_GridName = gridname + f'_d0{str(regrid_dom)}'
            MCIP_GridName_d01 = gridname + '_d01'
            terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/Post_combine.ok") == False:
                if CMAQcombine == False:  # 自己编写的Combine脚本，PM10还不能输出
                    terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/cctmCombine") # 整理结果文件
                    terminal_exec(terminal, f"cd {CMAQdata_dir}{MCIP_GridName}/cctm/") 
                    terminal_exec(terminal, f"mkdir toCombine")
                    terminal_exec(terminal, f"mv CCTM_ACONC_* toCombine")
                    terminal_exec(terminal, f"mv CCTM_APMDIAG_* toCombine")
                    time.sleep(5)
                    O3PM25Combine_CMAQfiles( 
                        CCTM_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctm/toCombine/",
                        Combine_file_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctmCombine/{MCIP_GridName}_combine_PM2503.nc",
                        GRIDDECfile_dir=f"{WCAS_dir}simulations/{gridname}/GRIDDESC",
                        start_time_yesterday=[1111,1,1] # 只是在nc文件中显示，可以忽略
                    )# 合并CMAQ文件
                else:   # 调用官方的combine脚本
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','APPL',ICON_APPL)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','GRID_NAME',MCIP_GridName)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','INPDIR',f'{CMAQdata_dir}{MCIP_GridName}/')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','METDIR',f'{CMAQdata_dir}{MCIP_GridName}/mcip')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','CCTMOUTDIR',f'{CMAQdata_dir}{MCIP_GridName}/cctm')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','POSTDIR',f'{CMAQdata_dir}{MCIP_GridName}/POST')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','START_DATE',cctm_start_date)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','END_DATE',cctm_end_date)
                    terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_combine_WCAS.csh {CMAQ_dir}POST/combine/scripts/")
                    time.sleep(0.8)
                    terminal_exec(terminal, f"cd {CMAQ_dir}POST/combine/scripts/")
                    terminal_exec(terminal, f"./run_combine_WCAS.csh")
                    time.sleep(2)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{CMAQ_dir}POST/combine/scripts/combinecsh.ok") == True:
                            break
            #D03：
            print(T.Now() + ' - ' + f'CombineCMAQ的结果... - d03')
            regrid_dom = 3
            MCIP_GridName = gridname + f'_d0{str(regrid_dom)}'
            MCIP_GridName_d01 = gridname + '_d01'
            terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/GRIDDESC {WCAS_dir}simulations/{gridname}/GRIDDESC")
            if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/Post_combine.ok") == False:
                if CMAQcombine == False:  # 自己编写的Combine脚本，PM10还不能输出
                    terminal_exec(terminal, f"mkdir {CMAQdata_dir}{MCIP_GridName}/cctmCombine") # 整理结果文件
                    terminal_exec(terminal, f"cd {CMAQdata_dir}{MCIP_GridName}/cctm/") 
                    terminal_exec(terminal, f"mkdir toCombine")
                    terminal_exec(terminal, f"mv CCTM_ACONC_* toCombine")
                    terminal_exec(terminal, f"mv CCTM_APMDIAG_* toCombine")
                    time.sleep(5)
                    O3PM25Combine_CMAQfiles( 
                        CCTM_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctm/toCombine/",
                        Combine_file_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctmCombine/{MCIP_GridName}_combine_PM2503.nc",
                        GRIDDECfile_dir=f"{WCAS_dir}simulations/{gridname}/GRIDDESC",
                        start_time_yesterday=[1111,1,1] # 只是在nc文件中显示，可以忽略
                    )# 合并CMAQ文件
                else:   # 调用官方的combine脚本
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','APPL',ICON_APPL)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','GRID_NAME',MCIP_GridName)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','INPDIR',f'{CMAQdata_dir}{MCIP_GridName}/')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','METDIR',f'{CMAQdata_dir}{MCIP_GridName}/mcip')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','CCTMOUTDIR',f'{CMAQdata_dir}{MCIP_GridName}/cctm')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','POSTDIR',f'{CMAQdata_dir}{MCIP_GridName}/POST')
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','START_DATE',cctm_start_date)
                    modify_csh_variable('namelists_cshfiles/run_combine_WCAS.csh','END_DATE',cctm_end_date)
                    terminal_exec(terminal, f"cp {WCAS_dir}namelists_cshfiles/run_combine_WCAS.csh {CMAQ_dir}POST/combine/scripts/")
                    time.sleep(0.8)
                    terminal_exec(terminal, f"cd {CMAQ_dir}POST/combine/scripts/")
                    terminal_exec(terminal, f"./run_combine_WCAS.csh")
                    time.sleep(2)
                    while (True):
                        time.sleep(1)
                        if os.path.exists(f"{CMAQ_dir}POST/combine/scripts/combinecsh.ok") == True:
                            break
            print(T.Now() + ' - ' + f'CMAQ结果Combine完成')
            terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/Post_combine.ok")

    
    


    if runPostprocess == True :print(T.Now() + ' - ' + f'计算输出逐小时、日均、模拟时段平均气象场...')
    terminal_exec(terminal, f"cd {WRF_dir}run/") #整理结果文件
    terminal_exec(terminal, f"cd WRFoutput") 
    terminal_exec(terminal, f"mkdir d01") 
    terminal_exec(terminal, f"mkdir d02") 
    terminal_exec(terminal, f"mkdir d03") 
    time.sleep(1.5)
    terminal_exec(terminal, f"mv wrfout_d01* d01") 
    terminal_exec(terminal, f"mv wrfout_d02* d02") 
    terminal_exec(terminal, f"mv wrfout_d03* d03") 
    terminal_exec(terminal, f"rm -rf wrfrst*") # 模拟完成，删除rst文件和bdy文件
    terminal_exec(terminal, f"rm -rf wrfinput*") # 模拟完成，删除rst文件和bdy文件
    time.sleep(4)
    terminal_exec(terminal, f"cd ..") 
    terminal_exec(terminal, f"mv WRFoutput {WCAS_dir}simulations/{gridname}/") 
    time.sleep(5)
    if dom_now_wind == True: WRFoutd02_files_dir_WRF = f"{WCAS_dir}simulations/{gridname}/WRFoutput/d0{str(regrid_dom)}/"
    else: WRFoutd02_files_dir_WRF = f"{WCAS_dir}simulations/{gridname}/WRFoutput/d0{str(regrid_dom-1)}/"
    if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/Post_WRFout.ok") == False and runPostprocess == True:
        WRFdata_output(
            WRFoutd03_files_dir=f"{WCAS_dir}simulations/{gridname}/WRFoutput/d0{str(regrid_dom)}/", # 在regriddom的上一层作为画风场的数据
            WRFoutd02_files_dir=WRFoutd02_files_dir_WRF,
            WCAS_output_dir=f"{WCAS_dir}simulations/{gridname}/outpics/",
            start_date=start_date_WRF+'-0'
        )
    if runPostprocess == True :print(T.Now() + ' - ' + f'气象场结果图输出完成')
    terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/Post_WRFout.ok")

    if runCMAQ:
        if runPostprocess == True :print(T.Now() + ' - ' + f'计算输出O3/PM25逐小时、日均、模拟时段平均近地面浓度分布...')
        terminal_exec(terminal, f"cd {CMAQdata_dir}{MCIP_GridName}/mcip")
        terminal_exec(terminal, f"mkdir temp")
        terminal_exec(terminal, f"cp GRIDCRO2D* temp")
        time.sleep(1)
        GRIDCRO2Df = os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/mcip/temp/")[0]
        terminal_exec(terminal, f"cp {CMAQdata_dir}{MCIP_GridName}/mcip/temp/{GRIDCRO2Df} {WCAS_dir}simulations/{gridname}/GRIDCRO2D.nc") # 获得用于CMAQ画图的GRIDCRO2D
        time.sleep(0.2)
        if dom_now_wind == True: WRFoutd02_files_dir_CMAQ = f"{WCAS_dir}simulations/{gridname}/WRFoutput/d0{str(regrid_dom)}/"
        else: WRFoutd02_files_dir_CMAQ = f"{WCAS_dir}simulations/{gridname}/WRFoutput/d0{str(regrid_dom-1)}/"
        if os.path.exists(f"{WCAS_dir}simulations/{gridname}/flagfiles/Post_CMAQout.ok") == False and runPostprocess == True:
            CMAQdata_output(
                CombineFile_dir=f"{CMAQdata_dir}{MCIP_GridName}/cctmCombine/{MCIP_GridName}_combine_PM2503.nc",
                GRIDCRO2D_dir=f"{WCAS_dir}simulations/{gridname}/GRIDCRO2D.nc",
                WRFoutd02_files_dir=WRFoutd02_files_dir_CMAQ,
                WCAS_output_dir=f"{WCAS_dir}simulations/{gridname}/outpics/",
                daynum = len(os.listdir(f"{CMAQdata_dir}{MCIP_GridName}/cctm/toCombine/"))/2,
                start_date=start_date_MCIP+'-0'
            )
        if runPostprocess == True :print(T.Now() + ' - ' + f'环境场结果图输出完成！')
        terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/Post_CMAQout.ok")

    print(T.Now() + ' - ' +f'WCAS运行完成 模拟过程名: {gridname} 运行耗时: {time.time() - WCAS_start_time}')
    terminal_exec(terminal, f"touch {WCAS_dir}simulations/{gridname}/flagfiles/WCAS.ok")


    








    

