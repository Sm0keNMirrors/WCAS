#!/bin/csh
#
# MET2MGN v2.10 
# --
#
#
# TPAR2IOAPI v2.03a 
# --added 26-category landuse capability for mm5camx (number of landuse categories defined by NLU) 
# --added capability for LATLON and UTM projections
# --added capability for MCIP v3.3 input (2m temperatures)
# --bug in PAR processing subroutine fixed where first few hours in GMT produced zero PAR
# --added code to fill missing par data (if valid data exists for the hours surrounding it)
#
# TPAR2IOAPI v2.0
# --added capability for MM5 or MCIP input
# 
#
#        RGRND/PAR options:
#           setenv MM5RAD  Y   Solar radiation obtained from MM5
#           OR 
#           setenv MCIPRAD Y   Solar radiation obtained from MCIP
#                  --MEGAN will internally calculate PAR for each of these options and user needs to  
#                    specify `setenv PAR_INPUT N' in the MEGAN runfile
#           OR
#           setenv SATPAR Y (satellite-derived PAR from UMD GCIP/SRB files)
#                  --user needs to specify `setenv PAR_INPUT Y' in the MEGAN runfile
#
#        TEMP options:
#           setenv CAMXTEMP Y         2m temperature, calculated from mm5camx output files
#           OR
#           setenv MM5MET  Y         2m temperature, calculated from MM5 output files
#                                     Note: 2m temperature is calculated since the P-X/ACM PBL
#                                     MM5 configuration (most commonly used LSM/PBL scheme for AQ 
#                                     modeling purposes) does not produce 2m temperatures.
#           OR
#           setenv MCIPMET Y         temperature obtained from MCIP
#              -setenv TMCIP  TEMP2   2m temperature, use for MCIP v3.3 or newer
#              -setenv TMCIP  TEMP1P5 1.5m temperature, use for MCIP v3.2 or older
#
#        TZONE   time zone for input mm5CAMx files 
#        NLAY    number of layers contained in input mm5CAMx files 
#        NLU     number of landuse categories contained in CAMx landuse file 
#

############################################################


# source ../setcase.csh
############################################################
# Episodes
############################################################
#set dom = $DOM 
#set STJD = 
#set EDJD = 

#setenv EPISODE_SDATE 2016001
#setenv EPISODE_STIME  000000    

############################################################
#set for grid
############################################################
#setenv GRIDDESC $MGNRUN/GRIDDESC
#setenv GDNAM3D d2


############################################################
# Setting up directories and common environment variable
############################################################
#source /home/Hmin/MEGANv2.10/setcase.csh

setenv PROG met2mgn
setenv EXE $MGNEXE/$PROG


set logdir = logdir/$PROG
if ( ! -e $logdir) mkdir -p $logdir

set INPPATH     = $MCIPOUT
set OUTPATH     = $MGNINP/MGNMET
if (! -e $OUTPATH) mkdir $OUTPATH


#setenv PFILE $OUTPATH/PFILE
#rm -fv $PFILE

############################################################
# Looping
############################################################
set JDATE = $STJD
while ($JDATE <= $EDJD)

setenv EPISODE_SDATE $JDATE
setenv EPISODE_STIME  000000
@ jdy  = $JDATE - 2000000

#set start/end dates
setenv STDATE ${jdy}00
setenv ENDATE ${jdy}24

#TEMP/PAR input choices
#
#set if using MM5 output files
setenv MM5MET N
setenv MM5RAD N
#setenv numMM5 2
#setenv MM5file1 /pete/pete5/fcorner/met/links/MMOUT_DOMAIN1_G$Y4$MM$DD
#setenv MM5file2 /pete/pete5/fcorner/met/links/MMOUT_DOMAIN1_G$Y4$MM$DD

#set if using UMD satellite PAR data
#set PARDIR = $MGNINP/PAR
#setenv SATPAR N
#set satpar1 = "$PARDIR/$Y2m1${MMm1}par.h"
#set satpar2 = "$PARDIR/$Y2${MM}par.h"

#if ($satpar1 == $satpar2) then
 # setenv numSATPAR 1
 # setenv SATPARFILE1 $satpar2
#else
#  setenv numSATPAR 2
#  setenv SATPARFILE1 $satpar1
#  setenv SATPARFILE2 $satpar2
#endif

#set if using MCIP output files
setenv MCIPMET Y
setenv TMCIP  TEMP2          #MCIP v3.3 or newer
#setenv TMCIP  TEMP1P5       #MCIP v3.2 or older

setenv MCIPRAD Y 
if ($JDATE == $EPISODE_SDATE) then
  setenv METCRO2Dfile1 $INPPATH/METCRO2D_$JDATE.nc
else
  setenv METCRO2Dfile1 $INPPATH/METCRO2D_$JDATE.nc
  setenv METCRO2Dfile2 $INPPATH/METCRO2D_$JDATE.nc
endif
setenv METCRO3Dfile  $INPPATH/METCRO3D_$JDATE.nc
setenv METDOT3Dfile  $INPPATH/METDOT3D_$JDATE.nc

setenv PFILE $OUTPATH/PFILE
 rm -rf $PFILE

setenv OUTFILE $OUTPATH/MET.MEGAN.$GDNAM3D.$JDATE.ncf
rm -rf $OUTFILE

$EXE |tee $logdir/log.$PROG.$GDNAM3D.$JDATE.txt 

@ JDATE += 1
echo $JDATE
end  # End while JDATE
