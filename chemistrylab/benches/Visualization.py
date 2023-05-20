from chemistrylab.benches.characterization_bench import CharacterizationBench
import numpy as np
from numba import jit

from PIL import Image,ImageDraw,ImageFont

@jit(nopython=True)
def fill_line(arr,x1,y1,x2,y2,lw):
    dy=y2-y1
    dx=x2-x1
    m = (dx**2+dy**2)**0.5

    # Create bounding box
    yi = max(min(y1,y2)-lw,0)
    yf = min(max(y1,y2)+lw,arr.shape[0])
    xi = max(x1-lw,0)
    xf = min(x2+lw,arr.shape[1])
    
    for i in range(yi,yf):
        for j in range(xi,xf):
            A=arr[i][j][0]/255
            x = j-x1
            y = i-y1

            #//d is the dot product
            d=x*dx+y*dy
            #X,Y is (x,y) - proj_{dx,dy}(x,y)
            X = x-d*dx/m/m
            Y = y-d*dy/m/m       
            if (d<m*m):
                #//Same assignment here but with the arrow body
                A=min(A,max(-3*d/lw,1-(lw/3.2-(X*X+Y*Y)**0.5)/0.4))
            arr[i][j] = max(A,0)*255




class Visualizer():
    def __init__(self, char_bench):
        self.char_bench = char_bench
        self.viz=dict(
            spectra=self.render_spectra,
            layers=self.render_layers,
            PVT=self.render_PVT,
        )

        self.h=360
        self.targets = {t:self.render_target(t) for t in char_bench.targets}

        # Pre-render the text for PVT information
        pvt = Image.new("RGBA", (self.h//2,self.h), (255,255,255))
        draw = ImageDraw.Draw(pvt)
        font = ImageFont.truetype("arial.ttf", self.h//8)
        draw.text((self.h//15, 0), "T  V  P", (0,0,0), font=font)
        self.i=0
        self.pvt=[np.asarray(pvt)[:,:,:3].copy() for k in range(char_bench.n_vessels)]
        
    def get_rgb(self, vessels):
        obs_list = self.char_bench.observation_list
        info = [self.targets[self.char_bench.target]] if "targets" in obs_list else []
        for v in vessels:
            info+= [self.viz[s](v) for s in obs_list if s in self.viz]
        return np.concatenate(info,axis=1)
    def render_layers(self,vessel):
        im = np.zeros([120,60,3],dtype=np.uint8)+255

        im[10:115,5:55] = 0
        im[10:110,10:50] = vessel.get_layers()[::-1,None,None]*255
        im[10:110,10:50,1:] //= 2

        im[10:110,10:50,1:]+=100
        
        div = self.h//120
        return im.repeat(div, axis=0).repeat(div, axis=1)

    def render_spectra(self,vessel):
        spectrum = np.clip(self.char_bench.get_spectra(vessel),0,0.99)

        im = np.zeros([self.h,self.h,3],dtype=np.uint8)+255
        offset=self.h//12
        max_height = self.h-2*offset
        scale = self.h//120
        for x in range(99):
            y0,y1 = int(spectrum[x]*max_height),int(spectrum[x+1]*max_height)

            fill_line(im[:,:,:2], x*scale+offset, max_height-y0+offset, (x+1)*scale+offset, max_height-y1+offset, scale+1)

            #im[max_height-y1+offset:max_height-y0+offset,x*scale+offset:(x+1)*scale+offset,:2] = 0
        #fill_line(im[:,:,:2],20,20,self.h-20,20,4)
        return im

    def render_PVT(self,vessel):
        
        im = self.pvt[self.i]
        self.i=(self.i+1)%self.char_bench.n_vessels
        offset=self.h//8
        w=self.h//12
        im[offset:]=255
        max_height = self.h-self.h//6

        t,v,p = 1-self.char_bench.encode_PVT(vessel)
        im[int(max_height*t)+offset:max_height+offset,0:w*2,1:] = 0
        im[int(max_height*v)+offset:max_height+offset,w*2:w*4,1::2] = 0
        im[int(max_height*p)+offset:max_height+offset,w*4:w*6,3:] = 0
        return im
    
    def render_target(self,target):
        h=self.h
        image = Image.new("RGBA", (h,h//6), (255,255,255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", h//10)
        draw.text((5, -2), target, (0,0,0), font=font)
        return np.asarray(image).transpose(1,0,2)[:,::-1,:3]
