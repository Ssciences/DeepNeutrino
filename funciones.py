### Dependences
import uproot
import numpy as np
import matplotlib.pyplot as plt
## from ROOT import TFile, TTree # use for saving funcion only
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

def reducir(number_of_files=int(10)): #izquierda sin defoult derecha con.
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

