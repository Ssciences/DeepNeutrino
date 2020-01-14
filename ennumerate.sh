# simple script to ennumerate and change the names of the files

echo insert path; # the path should be a directory containing root files
read path;
cd $path;
echo moving to $path
 
echo insert tipo; 
read tipo;
echo the type is ${tipo}.root

echo insert particle; 
read particle;
echo the particle is ${particle}.root
 

number_of_files=`ls -l *.root | wc`;
echo there are $number_of_files ROOT files
x=0 # initialize file number
y=1 # counter


# the programme
for i in *.root 
do 
mv $i ${particle}_${tipo}_$[x].root
x=$((x+y)) #update the variable x
done



