from chemistrylab.benches.characterization_bench import CharacterizationBench
import numpy as np
from numba import jit

from matplotlib import pyplot as plt
import io
import matplotlib.patches as mpatches


import matplotlib

RES = 1/2
matplotlib.rcParams.update({'font.size': 12*RES})

import matplotlib.style as mplstyle
mplstyle.use('fast')

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


def to_rgb(x):
  """Converts a 4-channel rgba image into a 3-channel rgb image"""
  # assume rgb premultiplied by alpha
  rgb, a = x[..., 1:4], x[...,0:1]
  return 255-a+rgb

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


class matplotVisualizer():
    def __init__(self, char_bench):
        self.char_bench = char_bench
        self.viz=dict(
            spectra=self.render_spectra,
            layers=self.render_layers,
            PVT=self.render_PVT,
        )

        self.heights = dict(
            spectra=2,
            layers=6,
            PVT=1
        )

        self.renders=[]
        self.w=4
        self.targets = {t:self.render_target(t) for t in char_bench.targets}
        self.steps=0

    def get_rgb(self, vessels):
        obs_list = self.char_bench.observation_list
        info = [a for a in obs_list if a!="targets"]
        heights = [self.heights[a] for a in info]
        row = len(info)
        col=len(vessels)

        if row*col==0:return np.zeros([0,0,3])
        first_render = not self.renders
        if first_render:
            self.fig,axs = fig,axs = plt.subplots(figsize=(col*self.w*RES, sum(heights)*RES), nrows=row, ncols=col, height_ratios=heights,dpi=100)
            self.axs=[axs] if row*col==1 else axs.flatten()
            self.renders=[None]*(row*col)
        else:
            self.fig.canvas.restore_region(self.bg)

        for j,v in enumerate(vessels):
          for i,func in enumerate(info):
            ax=self.axs[i*col+j]
            result = self.viz[func](v,ax, first = j==0, prev= self.renders[i*col+j])
            self.renders[i*col+j]=result

        if first_render:
            self.fig.tight_layout()
            self.bg = fig.canvas.copy_from_bbox(fig.bbox)

        #stack overflow said to do this to cast to array
        with io.BytesIO() as buff:
            self.fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))

        self.steps+=1

        return im[...,:3]#to_rgb(im)
            
    def render_layers(self,vessel,ax, first = False,prev=None):
        cmap='cubehelix'
        layers = vessel.get_layers()

        first=first and self.steps%100==0
        if prev is None or first:ax.clear()
        if first:
            cvals = (np.array([mat._color for mat in vessel._layer_mats]+[0.65])+0.2)%1
            im = ax.imshow([cvals],cmap=cmap,vmin=0,vmax=1)
            colors = [ im.cmap(im.norm(value)) for value in cvals]
            patches = [ mpatches.Patch(color=colors[i],label= name) for i,name in enumerate(list(vessel._layer_mats)+["air"]) ] 
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        

        if prev is None or first:
            prev = ax.imshow((np.stack([layers]).T[::-1] +0.2)%1,vmin=0,vmax=1,cmap=cmap,animated=True,aspect=0.025)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(str(vessel))
        else:
            prev.set_array((np.stack([layers]).T[::-1] +0.2)%1)

        return prev

    def render_spectra(self,vessel,ax, first = False, prev=None):

        
        spectrum = np.clip(self.char_bench.get_spectra(vessel),0,0.99)

        if prev is None:
            ax.clear()
            ax.set_xlabel("Wavelength (nm)")
            x=np.linspace(2000,20000,200)
            ax.set_xlim(2000,20000)
            ax.set_xticks([0,8000,16000])
            ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
            ax.set_ylim(0,1.05)
            ax.set_ylabel("Absorbance")
            prev = ax.plot(x,spectrum,alpha=1.0,lw=1)
        else:
            prev[0].set_ydata(spectrum)
        return prev

    def render_PVT(self,vessel,ax, first = False , prev=None):

        t,v,p = self.char_bench.encode_PVT(vessel)
        if prev is None:
            ax.clear()
            bt=ax.barh(0,t,color="r")
            bv=ax.barh(1,v)
            bp =ax.barh(2,p)
            ax.set_xlim(0,1)
            ax.set_xticks([])
            ax.set_yticks([0,1,2],["Temperature","Volume","Pressure"])

            prev=(bt[0],bv[0],bp[0])
        else:
            bt,bv,bp=prev
            bt.set_width(t)
            bv.set_width(v)
            bp.set_width(p)
        
        return prev
    
    def render_target(self,target):

        return

Visualizer = matplotVisualizer