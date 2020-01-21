#### Script to saving the images, 

# run this script in order to reduce the images and save them. As you can see, it is self explanatory. 

## imports 
from funciones import reducir_guardar

number_of_files=20 # number of files of each particle type. 
particle=['electron' , 'muon'] # different types of particles. 

for i in particle: # loop over particle
    print('we have lots of particles:{}'.format(particle))
    print('reducing particle type {}'.format(i))
    reducir_guardar(number_of_files, i) # this function will loop over files and events. 
    





