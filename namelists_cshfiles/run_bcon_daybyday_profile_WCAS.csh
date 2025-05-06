#!/bin/csh -f
rm -rf bconcsh.ok

# ======================= BCONv5.3.X Run Script ======================== 
# Usage: run.bcon.csh >&! bcon.log &                                
#
# To report problems or request help with this script/program:        
#             http://www.cmascenter.org
# ==================================================================== 

# ==================================================================
#> Runtime Environment Options
# ==================================================================

#> Choose compiler and set up CMAQ environment with correct 
#> libraries using config.cmaq. Options: intel | gcc | pgi
 
 setenv compiler gcc #编译器名称，我的是intel，你们用的应该是gcc

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
 set VRSN     = v532                    #> Code Version
set BCTYPE = profile
set start_time = 2024-12-14
set end_time = 2024-12-16
 
#> Horizontal grid definition 
setenv GRID_NAME CDsvSA15_d01
setenv CMAQdata_dir /home/shiroko/WCAS/simulations/CDsvSA15/
#setenv GRIDDESC $CMAQ_DATA/$APPL/met/mcip/GRIDDESC #> grid description file 
 setenv GRIDDESC ${CMAQdata_dir}${GRID_NAME}/mcip/GRIDDESC #mcip输出的GRIDDESC文件
 setenv IOAPI_ISPH 20                     #> GCTP spheroid, use 20 for WRF-based modeling
# =====================================================================
#> Input/Output Directories
# =====================================================================
 set MCIPDIR  = ${CMAQdata_dir}${GRID_NAME}/mcip       #mcip结果文件路径
 set OUTDIR   = ${CMAQdata_dir}${GRID_NAME}/bcon       #bcon输出路径
#> Set the build directory:
 set BLD      = ${CMAQ_HOME}/PREP/bcon/scripts/BLD_BCON_${VRSN}_${compilerString}
 set EXEC     = BCON_${VRSN}.exe  
 cat $BLD/BCON_${VRSN}.cfg; echo " "; set echo
#> I/O Controls
 setenv IOAPI_LOG_WRITE F     #> turn on excess WRITE3 logging [ options: T | F ]
 setenv IOAPI_OFFSET_64 YES   #> support large timestep records (>2GB/timestep record) [ options: YES | NO ]
 setenv EXECUTION_ID $EXEC    #> define the model execution id
 
 set start_time_s=`date -d $start_time +%s`
 set end_time_s=`date -d $end_time +%s`
 
 while ($end_time_s > $start_time_s)
 	 set previous_time=`date -d "$start_time -1 days" +%Y-%m-%d`
	 set next_time=`date -d "$start_time +1 days" +%Y-%m-%d`  
	 set APPL=`date -d $start_time +%Y%j`
	 echo $APPL 
	
	# =====================================================================
	#> BCON Configuration Options
	#
	# BCON can be run in one of two modes:                                     
	#     1) regrids CMAQ CTM concentration files (BC type = regrid)     
	#     2) use default profile inputs (BC type = profile)
	# =====================================================================
	
	 setenv BCON_TYPE ` echo $BCTYPE | tr "[A-Z]" "[a-z]" `
	
	# =====================================================================
	#> Input Files
	#  
	#  Regrid mode (BC = regrid) (includes nested domains, windowed domains,
	#                             or general regridded domains)
	#     CTM_CONC_1 = the CTM concentration file for the coarse domain          
	#     MET_CRO_3D_CRS = the MET_CRO_3D met file for the coarse domain
	#     MET_BDY_3D_FIN = the MET_BDY_3D met file for the target nested domain
	#                                                                            
	#  Profile mode (BC type = profile)
	#     BC_PROFILE = static/default BC profiles 
	#     MET_BDY_3D_FIN = the MET_BDY_3D met file for the target domain 
	#
	# NOTE: SDATE (yyyyddd), STIME (hhmmss) and RUNLEN (hhmmss) are only 
	#       relevant to the regrid mode and if they are not set,  
	#       these variables will be set from the input MET_BDY_3D_FIN file
	# =====================================================================
	#> Output File
	#     BNDY_CONC_1 = gridded BC file for target domain
	# =====================================================================
	 
	    set DATE = $start_time
	    set YYYYJJJ  = `date -ud "${DATE}" +%Y%j`   #> Convert YYYY-MM-DD to YYYYJJJ
	    set YYMMDD   = `date -ud "${DATE}" +%y%m%d` #> Convert YYYY-MM-DD to YYMMDD
	    set YYYYMMDD = `date -ud "${DATE}" +%Y%m%d` #> Convert YYYY-MM-DD to YYYYMMDD
	#   setenv SDATE           ${YYYYJJJ}
	#   setenv STIME           000000
	#   setenv RUNLEN          240000
	
	 if ( $BCON_TYPE == regrid ) then 
	    setenv CTM_CONC_1 /work/MOD3EVAL/sjr/CCTM_CONC_v53_intel18.0_2016_CONUS_test_${YYYYMMDD}.nc
	    setenv MET_CRO_3D_CRS $MCIPDIR/METCRO3D_${YYYYJJJ}.nc
	    setenv MET_BDY_3D_FIN $MCIPDIR/METBDY3D_${YYYYJJJ}.nc
	    setenv BNDY_CONC_1    "$OUTDIR/BCON_${VRSN}_${APPL}_${BCON_TYPE}_${YYYYMMDD} -v"
	 endif
	
	 if ( $BCON_TYPE == profile ) then
	    setenv BC_PROFILE $BLD/avprofile_cb6r3m_ae7_kmtbr_hemi2016_v53beta2_m3dry_col051_row068.csv
	    setenv MET_BDY_3D_FIN $MCIPDIR/METBDY3D_${YYYYJJJ}.nc
	    setenv BNDY_CONC_1    "$OUTDIR/BCON_${VRSN}_${APPL}_${BCON_TYPE}_${YYYYMMDD} -v"
	 endif
	
	# =====================================================================
	#> Output File
	# =====================================================================
	 
	#>- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	
	 if ( ! -d "$OUTDIR" ) mkdir -p $OUTDIR
	
	 ls -l $BLD/$EXEC; size $BLD/$EXEC
	# unlimit
	# limit
	
	#> Executable call:
	 time $BLD/$EXEC
	
	 set start_time=`date -d "$start_time +1 days" +%Y-%m-%d`
	 set start_time_s=`date -d $start_time +%s`
 end
 
touch bconcsh.ok

 exit()

