#!/bin/csh

rm -rf megancsh.ok

set start_date = 2024-12-14
set end_date = 2024-12-15

setenv STJD  `date -d ${start_date} +"%Y%j"`
setenv EDJD `date -d ${end_date} +"%Y%j" `
echo "STJD = $STJD"
echo "EDJD = $EDJD"

setenv MGNHOME /home/shiroko/MEGAN/MEGAN210/

setenv MGNSRC $MGNHOME/src
setenv MGNLIB $MGNHOME/lib
setenv MGNEXE $MGNHOME/bin
setenv MGNRUN $MGNHOME/work
setenv MGNINP /home/shiroko/MEGAN/preMEGAN_output/
setenv MGNOUT $MGNHOME/Output
setenv MGNINT $MGNHOME/Output/INT
setenv MGNLOG $MGNHOME/work/logdir

############################################################
#set for grid
############################################################
setenv GDNAM3D CDsvSA15_d03
setenv dom 3
setenv PRJ $GDNAM3D

setenv CAMQdata_dir /home/shiroko/WCAS/simulations/CDsvSA15/
setenv MCIPOUT ${CAMQdata_dir}${GDNAM3D}/mcip
setenv GRIDDESC $MCIPOUT/GRIDDESC

if ( ! -e $MGNINP ) then
   mkdir -p $MGNINP
   mkdir -p $MGNINP
   mkdir -p $MGNINP
endif
if ( ! -e $MGNINT ) mkdir -p $MGNINT
if ( ! -e $MGNLOG ) mkdir -p $MGNLOG

cd  $MGNHOME/work

./run.txt2ioapi.continous.v210.csh
./run.met2mgn.continous.v210.csh
./run.emproc.continous.v210.csh
./run.mgn2mech.continous.v210.csh

touch $MGNHOME/megancsh.ok

echo "end time `date`"
