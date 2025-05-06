#! /bin/csh -f
########################################################################
## Common setups
#source ../setcase.csh

#setenv PRJ sc 
setenv DOM $dom

setenv PROMPTFLAG N
setenv PROG   txt2ioapi
setenv EXEDIR $MGNEXE
setenv EXEC   $EXEDIR/$PROG
#setenv GRIDDESC $MGNRUN/GRIDDESC
#setenv GDNAM3D d2 

## File setups
## Inputs
setenv EFSTXTF $MGNINP/EF210.csv
setenv PFTTXTF $MGNINP/PFT210.csv
setenv LAITXTF $MGNINP/LAI210.csv
## Outputs
setenv EFMAPS  $MGNINP/EFMAPS.${PRJ}${DOM}.ncf
setenv PFTS16  $MGNINP/PFTS16.${PRJ}${DOM}.ncf
setenv LAIS46  $MGNINP/LAIS46.${PRJ}${DOM}.ncf

## Run control
setenv RUN_EFS T       # [T|F]
setenv RUN_LAI T       # [T|F]
setenv RUN_PFT T       # [T|F]
########################################################################





## Run TXT2IOAPI
rm -f $EFMAPS $LAIS46 $PFTS16
if ( ! -e $MGNLOG/$PROG ) mkdir -p $MGNLOG/$PROG
$EXEC | tee $MGNLOG/$PROG/log.run.$PROG.${PRJ}${DOM}.txt
