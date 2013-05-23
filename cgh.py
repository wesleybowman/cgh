import numpy as np
import matplotlib.pyplot as plt
import time

def getImage(image):
    '''Get the initial image that one wants to make a CGH out of. '''
    img=plt.imread(image)

    return img

def normalizeImage(img):
    '''This is to make sure every input image is seen as the same by the
       program later on. This way, the image will be between 0 and 1.'''

    imgMax=np.amax(img)

    for (i,j),k in np.ndenumerate(img):
        img[i][j]=k/imgMax

    return img

def getParameters(a,b,c=1,d=0, e=0):
    '''Get initial parameters.
       a is wavelength in meters
       b is how many dots per inch is needed
       c is the dimensions of the hologram in inches
       d is the offset of the object points on the x-axis
       e is the offset of the object points on the y-axis'''

    wavelength = a
    sampling=0.0254/b #0.0254 meters is one inch
    dimensions = c
    xOffset = d
    yOffset = e

    return wavelength,sampling,dimensions,xOffset,yOffset

def constants():
    '''Defines some values that do not need to be modified. '''

    k=2*np.pi/wavelength

    holoRange = dimensions*0.0254/2
    ipx = np.arange((-1*holoRange),(holoRange+sampling),sampling)
    ipy = np.arange((-1*holoRange),(holoRange+sampling),sampling)

    ipx=np.reshape(ipx,(1,ipx.shape[0]))
    ipy=np.reshape(ipy,(1,ipy.shape[0]))

    ipxShape=ipx.shape[1]
    ipyShape=ipy.shape[1]

    film = np.zeros((ipxShape,ipyShape))
    film=film+0j

    return k,holoRange,ipx,ipy,ipxShape,ipyShape,film

def getHoloPara(a=0.03,b=0.03,c=2):
    '''Get the parameters needed for creating the hologram.
       a is width in meters
       b is height in meters
       c is depth in meters'''

    width = a
    height = b
    depth = c

    objPoint=np.array([[0,0,4]])

    return width,height,depth,objPoint

def getObjectpoints(img, width, height, depth,objectpoints):
    '''Figure out which points in the image will be used as source points. '''

    obj = np.double(img)
    objX=obj.shape[0]
    objY=obj.shape[1]

    thresh = 0.5
    row = 0

    for i in xrange(objX):
        for j in xrange(objY):

            if obj[i][j]<thresh:
                if row==0:
                    objectpoints=np.array([[(i-objX/2)*(width/objX),
                              (j-objY/2)*(height/objY),
                                depth]])
                else:
                    temp= np.array([[(i-objX/2)*(width/objX),
                          (j-objY/2)*(height/objY),
                            depth]])
                    objectpoints=np.vstack((objectpoints,temp))

                row+=1

    return objectpoints

def getComplexwave():
    '''Iterate through every source point, and calculate the complex wave
       contribution at each sampling pixel on the film.'''
    for o in xrange(objPointShape):
        print o+1

        for i in xrange(ipxShape):
            for j in xrange(ipyShape):
                dx=objectpoints[o][0] - ipx[0][i]
                dy=objectpoints[o][1] - ipy[0][j]
                dz=objectpoints[o][2]

                distance=np.sqrt(dx**2+dy**2+dz**2)
                complexwave=np.exp(1j*k*distance)

                film[i][j]=film[i][j]+complexwave

def plotHologram(hologram):
    '''Plotting the real part of the hologram, and using a grey scale color
       map'''

    plt.imshow(hologram.real,cmap=plt.cm.Greys_r)
    plt.colorbar()
    plt.show()


if __name__=='__main__':

    t1=time.time()

    #img=plt.imread('blackA.jpg')
    #img=plt.imread('dot.png')
    #img=plt.imread('twoPixels.png')
    #img=plt.imread('onePixels.png')
    #img=plt.imread('smallA.png')

    img=getImage('smallA.png')
    img=np.mean(img,2)

    img=normalizeImage(img)

    wavelength,sampling,dimensions,xOffset,yOffset=getParameters(632e-9,600,1)

    k,holoRange,ipx,ipy,ipxShape,ipyShape,film=constants()

    width,height,depth,objPoint=getHoloPara()

    objectpoints = getObjectpoints(img, width, height, depth,objPoint)

    '''To test the 3-D portion of this program, two images can be stacked
       at different depth values, so that when reconstructed, you can get
       two distinct images at different z values. To stack two images,
       use the following snippit of code.'''

    #img2=plt.imread('smallB.png')
    #img2=np.mean(img2,2)
    #depth2=2
    #
    #objectpoints2=getObjectpoints(img2, width, height, depth2,objPoint)
    #
    #objectpoints=np.vstack((objectpoints,objectpoints2))

    objPointShape=objectpoints.shape[0]

    #offset the x-axis by some amount
    objectpoints[:,1] = objectpoints[:,1]+xOffset
    #offset the y-axis by some amount
    objectpoints[:,0] = objectpoints[:,0]+yOffset

    print 'Computing...\n'
    print 'Hologram resolution = %d,%d \n' %(ipxShape,ipyShape)
    print 'Number of source points from image = %d \n' %objPointShape
    print 'Calculating hologram for source point:'

    getComplexwave()

    t2=time.time()
    print t2-t1

    plotHologram(film)

