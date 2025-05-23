#!/bin/csh -f
rm -rf iconcsh.ok
# ======================= ICONv5.3.X Run Script ========================
# Usage: run.icon.csh >&! icon.log &                                   
#
# To report problems or request help with this script/program:         
#             http://www.cmascenter.org
# ==================================================================== 

# ==================================================================
#> Runtime Environment Options
# ==================================================================

#> Choose compiler and set up CMAQ environment with correct 
#> libraries using config.cmaq. Options: intel | gcc | pgi
 
 setenv compiler gcc #编译器名称，我的是intel，你们的估计是gcc

#> Source the config_cmaq file to set the run environment
 pushd ../../../
 source ./config_cmaq.csh $compiler
 popd

#> Check that CMAQ_DATA is set:
 if ( ! -e $CMAQ_DATA ) then
    echo "   $CMAQ_DATA path does not exist"
    exit 1
 endif
 echo " "; echo " Input data path, CMAQ_DATA set to $CMAQ_DATA"; echo " "

#> Set General Parameters for Configuring the Simulation
 set VRSN     = v532                   #> Code Version
set APPL = 20241214
set ICTYPE = profile

#> Set the working directory:
 set BLD      = ${CMAQ_HOME}/PREP/icon/scripts/BLD_ICON_${VRSN}_${compilerString}
 set EXEC     = ICON_${VRSN}.exe  
 cat $BLD/ICON_${VRSN}.cfg; echo " "; set echo

#> Horizontal grid definition 
setenv GRID_NAME CDsvSA15_d01
setenv CMAQdata_dir /home/shiroko/WCAS/simulations/CDsvSA15/
 setenv GRIDDESC ${CMAQdata_dir}${GRID_NAME}/mcip/GRIDDESC #mcip输出的GRIDDESC文件
 setenv IOAPI_ISPH 20                     #> GCTP spheroid, use 20 for WRF-based modeling

#> I/O Controls
 setenv IOAPI_LOG_WRITE F     #> turn on excess WRITE3 logging [ options: T | F ]
 setenv IOAPI_OFFSET_64 YES   #> support large timestep records (>2GB/timestep record) [ options: YES | NO ]
 setenv EXECUTION_ID $EXEC    #> define the model execution id

# =====================================================================
#> ICON Configuration Options
#
# ICON can be run in one of two modes:                                     
#     1) regrids CMAQ CTM concentration files (IC type = regrid)     
#     2) use default profile inputs (IC type = profile)
# =====================================================================

 setenv ICON_TYPE ` echo $ICTYPE | tr "[A-Z]" "[a-z]" ` 

# =====================================================================
#> Input/Output Directories
# =====================================================================
 set MCIPDIR  = ${CMAQdata_dir}${GRID_NAME}/mcip #mcip结果文件路径
 set OUTDIR   = ${CMAQdata_dir}${GRID_NAME}/icon #icon输出路径

# =====================================================================
#> Input Files
#  
#  Regrid mode (IC = regrid) (includes nested domains, windowed domains,
#                             or general regridded domains)
#     CTM_CONC_1 = the CTM concentration file for the coarse domain          
#     MET_CRO_3D_CRS = the MET_CRO_3D met file for the coarse domain
#     MET_CRO_3D_FIN = the MET_CRO_3D met file for the target nested domain 
#                                                                            
#  Profile Mode (IC = profile)
#     IC_PROFILE = static/default IC profiles 
#     MET_CRO_3D_FIN = the MET_CRO_3D met file for the target domain 
#
# NOTE: SDATE (yyyyddd) and STIME (hhmmss) are only relevant to the
#       regrid mode and if they are not set, these variables will 
#       be set from the input MET_CRO_3D_FIN file
# =====================================================================
#> Output File
#     INIT_CONC_1 = gridded IC file for target domain
# =====================================================================

set DATE = 2024-12-14
    set YYYYJJJ  = `date -ud "${DATE}" +%Y%j`   #> Convert YYYY-MM-DD to YYYYJJJ
    set YYMMDD   = `date -ud "${DATE}" +%y%m%d` #> Convert YYYY-MM-DD to YYMMDD
    set YYYYMMDD = `date -ud "${DATE}" +%Y%m%d` #> Convert YYYY-MM-DD to YYYYMMDD
#   setenv SDATE           ${YYYYJJJ}
#   setenv STIME           000000

 if ( $ICON_TYPE == regrid ) then
    setenv CTM_CONC_1 /work/MOD3EVAL/sjr/CCTM_CONC_v53_intel18.0_2016_CONUS_test_${YYYYMMDD}.nc
    setenv MET_CRO_3D_CRS $MCIPDIR/METCRO3D_${YYYYJJJ}.nc
    setenv MET_CRO_3D_FIN $MCIPDIR/METCRO3D_${YYYYJJJ}.nc
    setenv INIT_CONC_1    "$OUTDIR/ICON_${VRSN}_${APPL}_${ICON_TYPE}_${YYYYMMDD} -v"
 endif

 if ( $ICON_TYPE == profile ) then
    setenv IC_PROFILE $BLD/avprofile_cb6r3m_ae7_kmtbr_hemi2016_v53beta2_m3dry_col051_row068.csv
    setenv MET_CRO_3D_FIN $MCIPDIR/METCRO3D_${YYYYJJJ}.nc
    setenv INIT_CONC_1    "$OUTDIR/ICON_${VRSN}_${APPL}_${ICON_TYPE}_${YYYYMMDD} -v"
 endif
 
#>- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 if ( ! -d "$OUTDIR" ) mkdir -p $OUTDIR

 ls -l $BLD/$EXEC; size $BLD/$EXEC
# unlimit
# limit

#> Executable call:
 time $BLD/$EXEC
touch iconcsh.ok
 exit() 

