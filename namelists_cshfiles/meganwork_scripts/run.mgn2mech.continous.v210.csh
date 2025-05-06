#! /bin/csh -f
########################################################################
#source ../setcase.csh
## Directory setups
#setenv PRJ sc 
setenv PROMPTFLAG N

# Program directory
setenv PROG   mgn2mech
setenv EXEDIR $MGNEXE
setenv EXE    $EXEDIR/$PROG

# Input map data directory
setenv INPDIR $MGNINP

# Intermediate file directory
setenv INTDIR $MGNOUT/INT

# Output directory
setenv OUTDIR $MGNOUT

# MCIP input directory
setenv METDIR $MGNINP/MGNMET

# Log directory
setenv LOGDIR $MGNLOG/$PROG
if ( ! -e $LOGDIR ) mkdir -p $LOGDIR
########################################################################

#set dom = 9
set JD = $STJD
while ($JD <=  $EDJD)
########################################################################
# Set up time and date to process
setenv SDATE $JD        #start date
setenv STIME 0
setenv RLENG 250000
setenv TSTEP 10000
########################################################################

########################################################################
# Set up for MECHCONV
setenv RUN_SPECIATE   Y    # run MG2MECH

setenv RUN_CONVERSION Y    # run conversions?
                           # run conversions MEGAN to model mechanism
                           # units are mole/s

setenv SPCTONHR       N    # speciation output unit in tonnes per hour
                           # This will convert 138 species to tonne per
                           # hour or mechasnim species to tonne per hour.
                           
# If RUN_CONVERSION is set to "Y", one of mechanisms has to be selected.
#setenv MECHANISM    RADM2
#setenv MECHANISM    RACM
#setenv MECHANISM    CBMZ
#setenv MECHANISM    CB05
setenv MECHANISM    CB6
#setenv MECHANISM    SOAX
#setenv MECHANISM    SAPRC99
#setenv MECHANISM    SAPRC99Q
#setenv MECHANISM    SAPRC99X
# Grid name
#setenv GDNAM3D d2 

# EFMAPS NetCDF input file
setenv EFMAPS  $INPDIR/EFMAPS.${PRJ}${dom}.ncf

# PFTS16 NetCDF input file
setenv PFTS16  $INPDIR/PFTS16.${PRJ}${dom}.ncf

# MEGAN ER filename
setenv MGNERS $INTDIR/ER.$GDNAM3D.${SDATE}.ncf

# Output filename
setenv MGNOUT $OUTDIR/MEGAN210.$GDNAM3D.$MECHANISM.$SDATE.ncf

########################################################################
## Run speciation and mechanism conversion
if ( $RUN_SPECIATE == 'Y' ) then
   rm -f $MGNOUT
   echo 111
   #$EXE | tee $LOGDIR/log.run.$PROG.$GDNAM3D.$MECHANISM.$SDATE.txt
   ${MGNHOME}src/MGN2MECH/mgn2mech | tee $LOGDIR/log.run.$PROG.$GDNAM3D.$MECHANISM.$SDATE.txt
   #/home/shiroko/MEGAN/MEGAN210/src/MGN2MECH/mgn2mech | tee $LOGDIR/log.run.$PROG.$GDNAM3D.$MECHANISM.$SDATE.txt

endif

@ JD++
end  # End while JD
