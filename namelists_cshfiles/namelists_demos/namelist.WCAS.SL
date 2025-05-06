&dirs
  WPS_dir = "/home/shiroko/WPS-4.1/"
  WRF_dir = "/home/shiroko/WRF-4.1.2/"
  CMAQ_dir = "/home/shiroko/CMAQ/CMAQv5.3/"
  premegan_dir = "/home/shiroko/MEGAN/prepmegan4cmaq/"
  premeganinput_dir = "/home/shiroko/MEGAN/preMEGAN_input/"
  premeganoutput_dir = "/home/shiroko/MEGAN/preMEGAN_output/"
  MEGAN_dir = "/home/shiroko/MEGAN/MEGAN210/"
  MEIAT_dir = "/mnt/d/MEIAT-CMAQ-1.4/"
  MEIAT_python_dir = "D:/anaconda3/envs/arcgispro-py3-2/python.exe"
/

&grid
  ratio=3
  res_d01=27000
  std_lat1_d01=20.0
  std_lat2_d01=40.0

  lat_min_d01=25.12
  lat_max_d01=36.21
  lon_min_d01=97.22
  lon_max_d01=111.97

  lat_min_d02=27.4
  lat_max_d02=33.8
  lon_min_d02=100.5
  lon_max_d02=107.8

  lat_min_d03=29.7
  lat_max_d03=32.40
  lon_min_d03=102.5
  lon_max_d03=105.5
/

&date
  start_date = "2024-05-03"
  end_date = "2024-05-03"
/

&control
  cores_win = 8
  cores_WRF = 16
  cores_CMAQ_col = 4
  cores_CMAQ_row = 4
  GridName = "3kmCDTest"
  CMAQsimtype = 'regrid'
  regrid_dom = 3

  MannualExtent = .true.
  Extent = 102.95,104.91,30.06,31.47
  
  testGeogrid = .false.
  
  dom_now_wind = .false.
  ManualGrid = .false.

  runCMAQ = .true.
  runMEGAN = .false.
  runISAM = .false.
  runPostprocess = .true.
/

&manualgrid
 i_parent_start    = 1,11,22,
 j_parent_start    = 1,6,20,
 e_we          = 43,67,55
 e_sn          = 36,76,91
 dx = 27000,
 dy = 27000,
 ref_lat   =  35.00,
 ref_lon   =  102.9,
 truelat1  =  20.0,
 truelat2  =  40.0,
 stand_lon =  104.41,
/