homo_dft=`grep -A 2 HOMO ozone_dft.out | tail -1`
homo_ki=`grep -A 2 HOMO ozone_ki.out | tail -1`
a0=`grep nkscalfact ozone_ki.in | awk '{print $3}'`
EN=`grep "total energy =" ozone_dft.out  | awk '{print $4}'`
ENm1=`grep "total energy =" ozone_dft_n-1.out  | awk '{print $4}'`
alpha=$(echo $homo_dft $homo_ki $EN $ENm1 | awk -v a0=$a0 '{print (($3-$4)*13.6057*2-$1)*a0/($2-$1)}')

echo homo_dft = $homo_dft
echo homo_ki = $homo_ki
echo alpha0= $a0
echo E[N]= $EN
echo E[N-1] = $ENm1
echo alpha = $alpha
