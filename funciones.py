### Dependences
import uproot
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TFile, TTree # use for saving funcion only
########functions to decrese the number of pixel of our detector images#######################


def maxpool(im, h, w):
    #dependences
    #####the imputs#####
    # im.shape must be a matrix of(width, height)
    # w and h are the output weight and height respectively .
    
#### preliminaries###
    h_step=im.shape[0]//h
    w_step=im.shape[1]//w
    #print("we have lost", (im.shape[1]%w)*(im.shape[0]%h), "pixels along the way")
    
    reduced_im=np.zeros((h,w)) ##the new reduced matrix is initialized with zeros
    
    
    ########The algorithm#########
    for i in range(0,h): #loop over h
        for j in range(0,w): #loop over w
            pool=im[i*h_step:h_step*(i+1),j*w_step:(j+1)*w_step]
            reduced_im[i,j]=np.max(pool)
            
            
    return reduced_im

def maxpoolmod(im, h, w):
    # observacions, modifications to account for issues regarding h_step ~ 1
    #####the imputs#####
    # im.shape must be a matrix of(width, height)
    # w and h are the output weight and height respectively .
    
#### preliminaries###
    h_step=im.shape[0]//h
    w_step=im.shape[1]//w
    #print("we have lost", (im.shape[1]%w)*(im.shape[0]%h), "pixels along the way")
    
    reduced_im=np.zeros((h,w)) ##the new reduced matrix is initialized with zeros
    extra_pixels=im.shape[0]-(im.shape[0]//h)*h
    #print(extra_pixels)
    loss_h = (im.shape[0]//h *h)/im.shape[0] ##the percented of the image that we will lose
    #print('we are loosing without processing', 1-loss_h ,'pixels')
    count=0
    ##########The algorithm##########
    for i in range(0,h): #loop over h
        for j in range(0,w): #loop over w
            r=np.random.uniform()
            #print(count,r,':',i,j)
            if r>extra_pixels/h/w and count<extra_pixels:
                pool=im[i*h_step+count:h_step*(i+1)+count,(j)*w_step:(j+1)*w_step]
                reduced_im[i,j]=np.max(pool)
            if r<extra_pixels/h/w and count<extra_pixels:
                pool=im[i*h_step+count:h_step*(i+1)+count,(j)*w_step:(j+1)*w_step]
                reduced_im[i,j]=np.max(pool)
                count=count+1     
    return reduced_im ,count

def reducir(number_of_files=int(10), particle='electron'): #izquierda sin defoult derecha con.
    #### imput is the number of files to consider
    
    #### useful variables #####
    EventsPerFile=int(500)
    NChannel = int(1280)
    Nticks = int(1667)
    w , h = NChannel , Nticks
    newsize = 319
    v = np.zeros((newsize,newsize,EventsPerFile*number_of_files))
    ## the program ###
    
    for i in range(0,number_of_files): ### i stands for the file_number
        # load the file
        ## select particle type
        if particle == 'electron':
            file = uproot.open("/scratch/deandres/MC/Electrons/reco/Electron_reco_{}.root".format(i))
            tree = file["analysistree"]["anatree"] 
            ADC = tree['RecoWaveform_ADC']
            
        if particle == 'muon':
            file = uproot.open("/scratch/deandres/MC/Muons/reco/Muon_reco_{}.root".format(i))
            tree = file["analysistree"]["anatree"] 
            ADC = tree['RecoWaveform_ADC']
        
        
        
        
        for j in range(0,EventsPerFile): ## loop over events
            basketcache={} # reset the memory 
            lazy=ADC.lazyarray(basketcache=basketcache) # now the memory used is in the variable basketcach
            todo=lazy[j].reshape((w,h))
            v1=todo[0:newsize,:] #### For now, we want only the first view
            #v = maxpool(v1,newsize,newsize)
            v[:,:,EventsPerFile*i+j] = maxpool(v1,newsize,newsize)
            print( 'este es el cache:', basketcache.keys())
            print('reducing event number {} out of {}'.format(EventsPerFile*i+j,EventsPerFile*number_of_files))
    #### output is the rediced images as a numpy array
    return v

def reducirone(file_number=int(1), particle='electron'): #izquierda sin defoult derecha con.
    #### imput is the number of files to consider
    
    #### useful variables #####
    EventsPerFile=int(50)
    NChannel = int(1280)
    Nticks = int(1667)
    w , h = NChannel , Nticks
    newsize = 100
    v = np.zeros((newsize,newsize,EventsPerFile))
    ## the program ###
    

        # load the file
        ## select particle type
    if particle == 'electron':
        file = uproot.open("/scratch/deandres/MC/alongZ_2_3GeV/Electrons/raw/Electron_raw_{}.root".format(file_number))
        tree = file["analysistree"]["anatree"] 
        ADC = tree['RawWaveform_ADC']
            
    if particle == 'muon':
        file = uproot.open("/scratch/deandres/MC/Muons/reco/Muon_reco_{}.root".format(file_number))
        tree = file["analysistree"]["anatree"] 
        ADC = tree['RawWaveform_ADC']
        
        
        
        
    for j in range(0,EventsPerFile): ## loop over events
        basketcache={} # reset the memory 
        lazy=ADC.lazyarray(basketcache=basketcache) # now the memory used is in the variable basketcach
        todo=lazy[j].reshape((w,h))
        v2=todo[319:,:] #### For now, we want only the first view
        #v = maxpool(v1,newsize,newsize)
        v[:,:,j] = maxpool(v2,newsize,newsize)
        #print( 'este es el cache:', basketcache.keys())
        #print('reducing event number {} out of {}'.format(j,EventsPerFile))
    #### output is the rediced images as a numpy array
    return v

def reduciroriginal(number_of_files=int(10)): #izquierda sin defoult derecha con.
    #### imput is the number of files to consider
    ###initialize
	newsize = 279
	v = np.zeros((newsize,newsize,100*number_of_files))

	for i in range(0,number_of_files):

		print("reducing file{} ".format(i))

		file = uproot.open("{}-RecoFull-Parser.root".format(i))
		tree = file["analysistree"]["anatree"] 
		ADC = tree.array( b'RecoWaveform_ADC')
		NChannel = tree.array(b'RecoWaveforms_NumberOfChannels')
		Nticks = tree.array(b'RecoWaveform_NumberOfTicksInAllChannels')
		NTracks = tree.array(b'NumberOfTracks')
		
        #print(w,h)

        ###here a loop over files
		
		
      
         
		for j in range(0,100):
			w , h = int(NChannel[j]) , int(Nticks[j]/NChannel[j])
			if ADC[j].shape[0] == w*h:
				todo = ADC[j].reshape((w,h))
				v1 = todo[0:279,:]
				v[:,:,100*i+j] = maxpool(v1,newsize,newsize)
			
			## The future algorithm will include v2 as well. 
				
    #### outputs are the reduced images as a numpy array
	return v

##########I/O############
def guardar(v,filename):
    ##########description#########
    # this function stores the np vector 'v' into a 'root file'. Due to memory reasons, the other options are not
    # as good as root files. 
    
    ###### note ########
    #### it may be worth trying hdf5 files #########
    
    ###FUNCTION####
    ######dependences#####
    #from ROOT import TFile, TTree
    #import numpy as np
    a=v.flatten()
    f = TFile(filename, 'recreate')
    t = TTree('mytree', 'tree')
    
    
    t.Branch('im', a, 'myarray[{}]/D'.format(int(a.shape[0])))
    
    
    
    t.Fill()
    f.Write()
    f.Close()
    ##### output is None, this function will save the reduced images as a root file, it is probably not 
    ##### convinient to use the array. 
    return None    
    
def abrir(file,dim):
    ######description########
    #this function opens our root file, and gives us a numpy vector of the proper dimensions.
    
    file = uproot.open(file)
    tree=file[b'mytree;1']
    im=tree.array(b'im')
    im=im.reshape(dim)
    return im

def dibujar(event,im):
    fig = plt.figure(frameon = False)
    plt.imshow(im[:,:,event].T,cmap = 'jet',interpolation='none')
    fig.set_size_inches(5, 5) ##grey scale
    fig.show()
    
def reducir_guardar(number_of_files=1, particle='electron'):
    for i in range(number_of_files):
        print('file = {}'.format(i+1))
        v = reducirone(i, particle)
        if i<10:
            guardar(v,"r{}_0{}.root".format(particle,i))
        else: 
            guardar(v,"r{}_{}.root".format(particle,i))
            
def display(Event = 0,save = False):
    w=1280
    h=1667
    ### Open Wave form file
    file = uproot.open("/scratch/deandres/MC/alongZ_2_3GeV/Muons/raw/raw.root") ### you may have to change the path as you wish
    tree=file["analysistree"]["anatree"] 
    ADC=tree['RawWaveform_ADC'] # define the object ADC from the tree
    basketcache={}
    lazy=ADC.lazyarray(basketcache=basketcache)
    im=lazy[Event].reshape((w,h))
    v1=im[0:320,:]
    v2=im[320:,:]
    fig = plt.figure()
    plt.imshow(v1.T,cmap = 'jet', interpolation='none')
    fig.set_size_inches(10, 10)
    plt.ylabel("Ticks, drift time ")
    plt.xlabel("channel View1")
    plt.show()
    if save:
        plt.save("view1.png")
        
    plt.imshow(v2.T,cmap = 'jet', interpolation='none') ##grey scale
    fig.set_size_inches(10, 10)
    plt.ylabel("Ticks, drift time ")
    plt.xlabel("channel View2")
    plt.show()
    if save:
        plt.save("view2.png")
    
    ### open paraneters file. 
    En=[]
    x=[]
    y=[]
    z=[]
    theta=[]
    phi=[]
    file = uproot.open("/scratch/deandres/MC/alongZ_2_3GeV/Muons/gen/gen.root")
    tree = file["analysistree"]["anatree"]
    En = np.append(En,tree.array(b'MCTruth_Generator_StartEnergy').flatten())
    x = np.append(x,tree.array(b'MCTruth_Generator_StartPoint_X').flatten())
    y = np.append(y,tree.array(b'MCTruth_Generator_StartPoint_Y').flatten())
    z = np.append(z,tree.array(b'MCTruth_Generator_StartPoint_Z').flatten())
    theta = np.append(theta,tree.array(b'MCTruth_Generator_StartDirection_Theta').flatten())
    phi = np.append(phi,tree.array(b'MCTruth_Generator_StartDirection_Phi').flatten())

    
    
    
    
    
    print("The event is generated according to the following parameters")
    print("E = ",round(En[Event],2), 'GeV')
    print("-"*40)
    print('x0 = ',round(x[Event],2))
    print('y0 = ',round(y[Event],2))
    print('z0 = ',round(z[Event],2))
    print("-"*40)
    print('theta0 = ',round(theta[Event],2))
    print('phi0 = ',round(phi[Event],2))
    
    ## return the two images as numpy arrays
    return v1, v2 