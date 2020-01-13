# simple script to ennumerate and change the names of the files
number_of_files=`ls -l *.root | wc`;
echo there are $number_of_files ROOT files
x=0 # initialize file number
y=1 # counter

# the programme
for i in *.root 
do 
mv $i muon_reco_$[x].root
x=$((x+y))#update the variable x
done



