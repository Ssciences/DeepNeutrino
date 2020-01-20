#### Script to saving the images, 

### this substitudes the saving images of the preliminaries, due to the fact that my computer does not have Jupyter ROOT

## imports 
import uproot
import numpy as np
import matplotlib.pyplot as plt
from funciones import maxpool , dibujar 
from ROOT import TFile, TTree

def reducir(number_of_files=int(10)): #izquierda sin defoult derecha con.
    #### imput is the number of files to consider
    #### local variables #####
    EventsPerFile=int(500)
    NChannel = int(1280)
    Nticks = int(1667)
    w , h = NChannel , Nticks
    
    
    for i in range(0,number_of_files): ### i stands for the file_number
        # load the file
        file = uproot.open("/scratch/deandres/MC/Electrons/reco/Electron_reco_{0}.root".format(i))
        tree = file["analysistree"]["anatree"] 
        ADC = tree['RecoWaveform_ADC']
        basketcache={} 
        lazy=ADC.lazyarray(basketcache=basketcache) # now the memory used is in the variable basketcache
        
        
        #print(w,h)

        ###here a loop over files
        newsize = 319
        v = np.zeros((newsize,newsize,EventsPerFile*number_of_files))
        vred = v
        for j in range(0,EventsPerFile): ## loop over events
            basketcache={} # reset the memory 
            todo=lazy[j].reshape((w,h))
            v1=todo[0:newsize,:] #### For now, we want only the first view
            v[:,:,EventsPerFile*i+j] = maxpool(v1,newsize,newsize)
    #### output is the rediced images as a numpy array
    return v

