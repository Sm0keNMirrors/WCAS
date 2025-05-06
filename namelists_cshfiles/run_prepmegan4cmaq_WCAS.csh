#

rm -rf premegancsh.ok

./prepmegan4cmaq_lai.x < prepmegan4cmaq.inp 


./prepmegan4cmaq_pft.x < prepmegan4cmaq.inp 


./prepmegan4cmaq_ef.x < prepmegan4cmaq.inp

touch premegancsh.ok

