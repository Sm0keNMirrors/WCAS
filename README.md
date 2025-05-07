<<<<<<< HEAD
# WCAS
=======
### WCAS (WRF-CMAQ Auto Script)

基于控制参数和预设的模型输入参数自动地控制WRF-CMAQ模拟。

进行自动运行前，需确保模型预设输入参数能够手动地无误运行。

##### 脚本运行需要安装的模型：

WPS(v4.1)

WRF(v4.1.2/4.3.3/4.5.1)

CMAQ(v5.3/5.4)

MEGAN(v2.1)

MEIAT-CMAQ(v1.4) https://github.com/Airwhf/MEIAT-CMAQ

##### 使用方法：

① 使用包含相关库的python环境运行

②通过包装后的WCAS执行：

https://drive.google.com/file/d/1dxAfXq-O064GfZ8TE2EhK2Zq4Ew4sE2-/view?usp=drive_link

将其安装到WCAS所在目录，进入并执行：

```
./WCAS "$PWD/"
```

##### namelist.WCAS说明：

```
&dirs
  WPS_dir = "/home/shiroko/WPS-4.1/" #WPS所在目录
  WRF_dir = "/home/shiroko/WRF-4.5.1/" #WRF所在目录
  CMAQ_dir = "/home/shiroko/CMAQ/CMAQv5.3/"  #CMAQ所在目录
  premegan_dir = "/home/shiroko/MEGAN/prepmegan4cmaq/" #prepmegan4cmaq所在目录
  premeganinput_dir = "/home/shiroko/MEGAN/preMEGAN_input/" #prepMEGAN输入的ef pft 相关文件所在目录
  premeganoutput_dir = "/home/shiroko/MEGAN/preMEGAN_output/" #prepmegan4cmaq输出的csv文件所在目录
  MEGAN_dir = "/home/shiroko/MEGAN/MEGAN210/"  #MEGAN210的目录
  MEIAT_dir = "/mnt/d/MEIAT-CMAQ-1.4/" #MEIAT1.4所在的Windows目录，但要以mnt来表示
  MEIAT_python_dir = "D:/anaconda3/envs/arcgispro-py3-2/python.exe" #arcgispro的conda的python.exe所在目录,要以Windows的目录形式表示
  
/
```

```
&grid
  ratio=3 #分辨率比例
  res_d01=27000# 最外层分辨率
  std_lat1_d01=20.0 
  std_lat2_d01=40.0 

  lat_min_d01=25.63  #三层嵌套的各自经纬度范围
  lat_max_d01=35.21
  lon_min_d01=97.53
  lon_max_d01=108.75

  lat_min_d02=28.02
  lat_max_d02=32.57
  lon_min_d02=101.53
  lon_max_d02=106.30

  lat_min_d03=29.81
  lat_max_d03=31.16
  lon_min_d03=103.2
  lon_max_d03=104.8
/
```

```
&date 
#模拟并输出结果的时间段，以CAMQ模拟的环境场为基准，WRF的结果时间范围会前后多1天
  start_date = "2024-03-01"  
  end_date = "2024-03-10"
/
```

```
&control 
  cores_win = 16 MEIAT所使用的核心数
  cores_WRF = 8 WRF所使用的核心数，在模拟范围较小时，需要降低此核数，否则会模拟失败
  cores_CMAQ_col = 4 CMAQ的行列核数，总核数是两者相乘
  cores_CMAQ_row = 4
  GridName = "SLexample01" #此次模拟的名称，每次执行时需要更改，同时每次模拟的结果不会自动删除，WRF在							run的WRFoutput中，CMAQ在CMAQdata中，结果图在WCAS路径下
  CMAQsimtype = 'regrid' #模拟方式，分为profile和regrid
  							profile：以清洁边界场模拟，模拟时间较快。模拟结果对于O3结果误差较大，									PM2.5影响较小。
  							regrid：以最外层的浓度场作为边界场模拟，要进行第一层和目标层2次模拟，模								拟时间较长。模拟结果会更加准确
  regrid_dom = 3  # 输出的目标层是2还是第3层，
  regrid_D02andD03 = .false. # 对D02和D03是否都进行前一层的BCONregrid
  CCTMversion = '532' # CCTM的版本
  BCICversion = '532' # BCON ICON 的版本
  compiler_str = 'gcc' # 编译器str
  max_dom = 3 # 最大模拟嵌套数
  
  aria2c_download = .false. # 是否使用aria2c来下载再分析资料，前提下载环境需安装aria2c
  remote_server_download = .false. # 是否使用远程服务器下载再分析资料
  remoteServer_hostname = 'xx.xx.xxx.xxx' # 远程服务器地址
  remoteServer_port = 1120 # 远程服务器端口
  remoteServer_username = 'root' # 远程服务器登录用户名
  remoteServer_password = 'X' # 远程服务器密码
  ForceGFS = .False. # 是否下载强制使用GFS，先FNL在GFS
  
  MannualExtent = .true. # 结果图的经纬度范围是否手动指定，若为false，则直接嵌套层的经纬度范围
  Extent = 103.7,104.4,30.16,30.74 # 结果图的经纬度范围
  
  testGeogrid = .true.   # 是否在运行中测试geogrid的范围时候合理，并可以重新修改namelist.WCAS的								grid模块进行重新设置
  
  dom_now_wind = .true.   # 是否直接用当前dom的风速来绘制风场，当指定了较小的extent时，用上层风场可							能会导致风场箭头稀疏
  ManualGrid = .false. # 是否手动设定嵌套层参数，用于通过已存在的namelist.wps进行模拟，需在下方的							&manualgrid进行填写，若为false，则&manualgrid中的不会被使用
  
  runCMAQ = .false. # 是否运行CMAQ，若为false，则仅运行WRF，然后后处理中也仅输出WRF的结果
  runMEGAN = .true.    # 是否通过MEGAN进行生物源计算，会提高暖季O3模拟的结果准确性，对PM2.5的影响较							小
  runISAM = .false.	   # 是否打开ISAM模块，注意，ISAM运行需要的文件需手动准备(REGION.NC, 								CONTROL.TXT, EMISSCONTOL.NML)，且结果不会自动处理。
  runPostprocess = .true. # 是否进行后处理
  CMAQcombine = .false. # 是否直接使用POST工具对CMAQ输出结果进行combine
  
  Mannualterminal = .false. # 脚本运行后会自动打开terminal进行控制模拟，false则表示手动指定tty
  control_terminal = "/dev/pts/0" # Mannualterminal为true时采用，指定tty的路径号
/

```

```
&manualgrid  # 参数与WPS的input一致
i_parent_start = 1, 22, 81
    j_parent_start = 1, 17, 62
    e_we = 118, 169, 130
    e_sn = 126, 154, 112
    dx = 27000
    dy = 27000
    ref_lat = 35.0
    ref_lon = 105.0
    truelat1 = 25.0
    truelat2 = 47.0
    stand_lon = 105.0
/
```

```
&ISAMcontrol
  if_makemask = .true. # 是否让脚本自动根据输入文件来制作mask，否则需要手动准备mask的nc
  mask_shpfilenames = 'chengdu'  # 某个区域mask的shp文件名(不带拓展名)，每个区域shp只能有一个多边									形
  mask_varnames = 'CD', #区域shp文件对应的ISAM的mask变量名，不能多于三个字母，多个区域时								mask_shpfilenames的名称要与要mask_varnames对应
/
自动运行ISAM要自行准备并修改input/ISAM文件夹中的ISAM输入文件：
ISAM_CONTROL.txt，区域shp文件到maskshps文件夹，以及对应CMAQ版本的control.nml中的区域信息
若手动输入mask，需命名为ISAM_REGION.nc
```

##### 路径解释：

WCAS/input/ ：用于自动绘图的shp与字体输入、ISAM区域输入

WCAS/namelists_cshfiles/：模型的参数控制，用于脚本控制模型运行，若修改需保证参数无误

WCAS/tools/：脚本的一些外部功能代码

##### 其他：

其他需要修改的模拟参数可以WCAS中的namelist与csh文件进行对应的修改，但需保证相关参数化方案能在手动模拟时无误。

