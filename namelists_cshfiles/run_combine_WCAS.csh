#! /bin/csh -f

rm -rf combinecsh.ok

# ====================== COMBINE_v5.3.X Run Script =================== 
# Usage: run.combine.uncoupled.csh >&! combine.log &                                
#
# To report problems or request help with this script/program:     
#             http://www.epa.gov/cmaq    (EPA CMAQ Website)
#             http://www.cmascenter.org  (CMAS Website)
# ===================================================================  

# ==================================================================
#> Runtime Environment Options
# ==================================================================

#> Choose compiler andset up CMAQ environment with correct
#> libraries using config.cmaq. Options: intel | gcc | pgi
setenv compiler intel

 #> Source the config.cmaq file toset the build environment
 cd ../../..
 source ./config_cmaq.csh
       
#>set General Parameters for Configuring the Simulation
set VRSN = v533              #> Code Version
set PROC = mpi               #> serial or mpi
set MECH = cb6r3_ae6_aq      #> Mechanism ID
set APPL = 20240315        #> Application Name (e.g. Gridname)
set GRID_NAME = yajiang_wf
                                                      
#> Define RUNID as any combination of parameters above or others. By default,
#> this information will be collected into this one string, $RUNID, for easy
#> referencing in output binaries and log files as well as in other scripts.
set RUNID = ${VRSN}_${compilerString}_${APPL}

#>set the build directory if this was notset above 
#> (this is where the CMAQ executable is located by default).
 if ( ! $?BINDIR ) then
set BINDIR = $CMAQ_HOME/POST/combine/scripts/BLD_combine_${VRSN}_${compilerString}
 endif

#>set the name of the executable.
set EXEC = combine_${VRSN}.exe

#>set location of CMAQ repo.  This will be used to point to the correct species definition files.
set REPO_HOME = ${CMAQ_REPO}

setenv INPDIR  /home/shiroko/CMAQdata/$GRID_NAME
#>set working, input and output directories
set METDIR = $INPDIR/mcip            #> Met Output Directory
set CCTMOUTDIR = /home/shiroko/CMAQdata/$GRID_NAME/cctm      #> CCTM Output Directory
set POSTDIR = /home/shiroko/CMAQdata/$GRID_NAME/POST                      #> Location where combine file will be written

  if ( ! -e $POSTDIR ) then
	  mkdir $POSTDIR
  endif



# =====================================================================
#> COMBINE Configuration Options
# =====================================================================

#>set Start and End Days for looping
set START_DATE = "2024-03-15"     #> beginning date (July 1, 2016)
set END_DATE = "2024-03-17"     #> ending date    (July 14, 2016)
 
#>set location of species definition files for concentration and deposition species.
setenv SPEC_CONC $REPO_HOME/POST/combine/scripts/spec_def_files/SpecDef_${MECH}.txt
setenv SPEC_DEP $REPO_HOME/POST/combine/scripts/spec_def_files/SpecDef_Dep_${MECH}.txt

#> Use GENSPEC switch to generate a new specdef file (does not generate output file).
setenv GENSPEC N


# =====================================================================
#> Begin Loop Through Simulation Days to Create ACONC File
# =====================================================================

#>set the species definition file for concentration species.
setenv SPECIES_DEF $SPEC_CONC
 
#> Loop through all days between START_DAY and END_DAY
set TODAYG = ${START_DATE}
set TODAYJ = `date -ud "${START_DATE}" +%Y%j` #> Convert YYYY-MM-DD to YYYYJJJ
set STOP_DAY = `date -ud "${END_DATE}" +%Y%j` #> Convert YYYY-MM-DD to YYYYJJJ

 while ($TODAYJ <= $STOP_DAY )  #>Compare dates in terms of YYYYJJJ
 
  #> Retrieve Calendar day Information
set YYYY = `date -ud "${TODAYG}" +%Y`
set YY = `date -ud "${TODAYG}" +%y`
set MM = `date -ud "${TODAYG}" +%m`
set DD = `date -ud "${TODAYG}" +%d`
  #> for files that are indexed with Julian day:
set YYYYJJJ = `date -ud "${TODAYG}" +%Y%j` 

  #> Define name of combine output file to save hourly average concentration.
  #> A new file will be created for each month/year.
setenv OUTFILE ${POSTDIR}/COMBINE_ACONC_${RUNID}_$YYYY$MM.nc

  #> Define name of input files needed for combine program.
  #> File [1]: CMAQ conc/aconc file
  #> File [2]: MCIP METCRO3D file
  #> File [3]: CMAQ APMDIAG file
  #> File [4]: MCIP METCRO2D file
setenv INFILE1 $CCTMOUTDIR/CCTM_ACONC_${RUNID}_$YYYY$MM$DD.nc
setenv INFILE2 $METDIR/METCRO3D_$YYYYJJJ.nc
setenv INFILE3 $CCTMOUTDIR/CCTM_APMDIAG_${RUNID}_$YYYY$MM$DD.nc
setenv INFILE4 $METDIR/METCRO2D_$YYYYJJJ.nc

  #> Executable call:
   ${BINDIR}/${EXEC}

  #> Increment both Gregorian and Julian Days
set TODAYG = `date -ud "${TODAYG}+1days" +%Y-%m-%d` #> Add a day for tomorrow
set TODAYJ = `date -ud "${TODAYG}" +%Y%j` #> Convert YYYY-MM-DD to YYYYJJJ

 end #Loop to the next Simulation Day


# =====================================================================
#> Begin Loop Through Simulation Days to Create DEP File
# =====================================================================

#>set the species definition file for concentration species.
setenv SPECIES_DEF $SPEC_DEP
 
#> Loop through all days between START_DAY and END_DAY
set TODAYG = ${START_DATE}
set TODAYJ = `date -ud "${START_DATE}" +%Y%j` #> Convert YYYY-MM-DD to YYYYJJJ
set STOP_DAY = `date -ud "${END_DATE}" +%Y%j` #> Convert YYYY-MM-DD to YYYYJJJ

 while ($TODAYJ <= $STOP_DAY )  #>Compare dates in terms of YYYYJJJ
 
  #> Retrieve Calendar day Information
set YYYY = `date -ud "${TODAYG}" +%Y`
 set YY = `date -ud "${TODAYG}" +%y`
 set MM = `date -ud "${TODAYG}" +%m`
 set DD = `date -ud "${TODAYG}" +%d`
  #> for files that are indexed with Julian day:
set YYYYJJJ = `date -ud "${TODAYG}" +%Y%j` 

  #> Define name of combine output file to save hourly total deposition.
  #> A new file will be created for each month/year.
 setenv OUTFILE ${POSTDIR}/COMBINE_DEP_${RUNID}_$YYYY$MM

  #> Define name of input files needed for combine program.
  #> File [1]: CMAQ DRYDEP file
  #> File [2]: CMAQ WETDEP file
  #> File [3]: MCIP METCRO2D
  #> File [4]: {empty}
 setenv INFILE1 $CCTMOUTDIR/CCTM_DRYDEP_${RUNID}_$YYYY$MM$DD.nc
 setenv INFILE2 $CCTMOUTDIR/CCTM_WETDEP1_${RUNID}_$YYYY$MM$DD.nc
 setenv INFILE3 $METDIR/METCRO2D_$YYYYJJJ.nc
 setenv INFILE4

  #> Executable call:
   ${BINDIR}/${EXEC}

 set progstat = ${status}
   if ( ${progstat} ) then
     echo "ERROR ${progstat} in $BINDIR/$EXEC"
     exit( ${progstat} )
   endif

  #> Increment both Gregorian and Julian Days
 set TODAYG = `date -ud "${TODAYG}+1days" +%Y-%m-%d` #> Add a day for tomorrow
 set TODAYJ = `date -ud "${TODAYG}" +%Y%j` #> Convert YYYY-MM-DD to YYYYJJJ

 end #Loop to the next Simulation Day

 touch $CMAQ_HOME/POST/combine/scripts/combinecsh.ok

 exit()
