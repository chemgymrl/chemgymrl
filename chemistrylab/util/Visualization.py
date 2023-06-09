from chemistrylab.benches.characterization_bench import CharacterizationBench
from chemistrylab.vessel import Vessel
from typing import NamedTuple, Tuple, Callable, Optional, List

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

class numbaVisualizer():
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
        
    def get_rgb(self, vessels: Tuple[Vessel]):
        obs_list = self.char_bench.observation_list
        info = [self.targets[self.char_bench.target]] if "targets" in obs_list else []
        for v in vessels:
            info+= [self.viz[s](v) for s in obs_list if s in self.viz]
        return np.concatenate(info,axis=1)
    def render_layers(self, vessel: Vessel):
        im = np.zeros([120,60,3],dtype=np.uint8)+255

        im[10:115,5:55] = 0
        im[10:110,10:50] = vessel.get_layers()[::-1,None,None]*255
        im[10:110,10:50,1:] //= 2

        im[10:110,10:50,1:]+=100
        
        div = self.h//120
        return im.repeat(div, axis=0).repeat(div, axis=1)

    def render_spectra(self, vessel: Vessel):
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

    def render_PVT(self, vessel: Vessel):
        
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

    legend_update_delay=1

    def __init__(self, char_bench):
        self.char_bench = char_bench
        self.viz=dict(
            spectra=self.render_spectra,
            layers=self.render_layers,
            PVT=self.render_PVT,
            targets=self.render_target,
        )

        self.heights = dict(
            spectra=2,
            layers=6,
            PVT=1,
            targets=0.25,
        )

        self.renders=[]
        self.w=4
        self.steps=0

    def get_rgb(self, vessels: Tuple[Vessel]):
        obs_list = self.char_bench.observation_list
        info = obs_list#[a for a in obs_list if a!="targets"]
        heights = [self.heights[a] for a in info]
        row = len(info)
        col=len(vessels)

        if row*col==0:return np.zeros([0,0,3])
        first_render = not self.renders
        if first_render:
            self.fig,axs = fig,axs = plt.subplots(figsize=(col*self.w*RES, sum(heights)*RES), nrows=row, ncols=col, height_ratios=heights,dpi=100)
            self.axs=[axs] if row*col==1 else axs.flatten()
            self.renders=[None]*(row*col)
            plt.close()
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

    def render_target(self, vessel: Vessel, ax, first = False, prev = None):
        if not first:
            ax.axis("off")
            return None
        if prev is None:
            ax.clear()
            ax.axis("off")
            text = ax.text(0,0,'Target: '+self.char_bench.target)
        else:
            text=prev
            newtext = 'Target: '+self.char_bench.target
            if newtext!=prev.get_text():
                prev.set_text(newtext)
        return text

    def render_layers(self, vessel: Vessel, ax, first = False,prev=None):
        cmap='cubehelix'
        layers = vessel.get_layers()

        first=first and self.steps%matplotVisualizer.legend_update_delay==0
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

    def render_spectra(self, vessel: Vessel, ax, first = False, prev=None):

        
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

    def render_PVT(self, vessel: Vessel, ax, first = False , prev=None):

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

    @staticmethod
    def display_vessels(vessels: Tuple[Vessel],observation_list: Tuple[str]):
        characterization_bench = CharacterizationBench(observation_list,[""],len(vessels))        
        visual = Visualizer(characterization_bench)    
        heights = [visual.heights[a] for a in observation_list]
        row = len(observation_list)
        col=len(vessels)
        fig,axs = plt.subplots(figsize=(col*visual.w*RES, sum(heights)*RES), nrows=row, ncols=col, height_ratios=heights,dpi=100)
        axs=[axs] if row*col==1 else axs.flatten()
        for j,v in enumerate(vessels):
          for i,func in enumerate(observation_list):
            visual.viz[func](v,axs[i*col+j], first = j==0)
        fig.tight_layout()
        plt.show()

class pygameVisualizer():
    """
    Class to visualize the chemistry benches.
    
    The dictionary viz gives a list of observations the class can visualize. These should correspond
    to observations which come from the characterization bench.

    Call get_rgb(vessels) to get an rgb image of your vessel observations (which visualize all observations set in the
    characterization bench)

    The resultant image is tiled such that every row has one type of observation, and every column contains all observations
    for a single vessel.

    """
    def __init__(self, char_bench):
        """
        Args:
            char_bench (CharacterizationBench): Characterization bench with info on what observations we need.
        """
        self.char_bench = char_bench
        self.viz=dict(
            spectra=self.render_spectra,
            layers=self.render_layers,
            PVT=self.render_PVT,
            targets=self.render_target,
        )

        self.heights = dict(
            spectra=0.5,
            layers=1,
            PVT=0.25,
            targets=0.125,
        )
        self.w=480
        self.screen_height = sum(self.heights[a] for a in self.char_bench.observation_list if a in self.heights)*self.w
        global pygame, gfxdraw
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        pygame.init()

        self.targets = {t:self._prerender_text('Target: '+t,self.w/10) for t in char_bench.targets}
        self.misc_text = {x: self._prerender_text(x,self.w/20) for x in ["Wavelength","Absorbance"]}
        self.misc_text["Absorbance"] = pygame.transform.rotate(self.misc_text["Absorbance"],90)

    def get_rgb(self,vessels: Tuple[Vessel]):

        """
        Create an rgb image corresponding to the observations of each vessel.
        """

        obs_list = [a for a in self.char_bench.observation_list if a in self.viz]

        self.surf = pygame.Surface((self.w*len(vessels), self.screen_height))
        self.surf.fill((255, 255, 255))

        for j,v in enumerate(vessels):
          cur_height = 0
          for i,func in enumerate(obs_list):
            self.viz[func](v, j*self.w, cur_height)
            cur_height += self.heights[func]*self.w
        
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )


    def render_spectra(self, vessel: Vessel,  x: int, y:int):
        """
        Method to visualize the spectral information of a vessel.

        Args:
            vessel (Vessel): The vessel inputted for spectroscopic analysis.
            x (int): Position of the top left corner of this image tile
            y (int): Position of the top right corner of ths image tile
        """
        spectrum = np.clip( y + self.w*5/12*(1 - self.char_bench.get_spectra(vessel)), y+1, y+self.w*5/12-2)
        xs = np.linspace(0, 1, 100) * self.w*5/6 + x +self.w/12
        xys = list(zip(xs, spectrum))
        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(30, 30, 255))

        bbox = [(self.w/12+x,y),(self.w/12+x,y+self.w*5/12-1),(self.w*11/12+x,y+self.w*5/12-1),(self.w*11/12+x,y)]

        pygame.draw.lines(self.surf, points=bbox, closed=True, color=(0, 0, 0))

        self.surf.blit(self.misc_text["Wavelength"],(x+self.w*5/12, y+self.w*5/12))
        self.surf.blit(self.misc_text["Absorbance"],(x+self.w/24, y+self.w/12))

    def _prerender_text(self, text, fs):
        """
        Returns a render of the input text
        
        Args:
            text (str): The text to render
            fs (float): The font size of the text
        """
        font = pygame.font.SysFont(None, int(fs))
        img = font.render(text, True, (0, 0, 0))
        return img
    def render_target(self, vessel: Vessel, x: int, y:int):
        """
        Method to display the benches target.

        Args:
            vessel (Vessel): A vessel
            x (int): The target will only be rendered if x=0
            y (int): Position of the top right corner of ths image tile
        """
        if x>0:return
        img = self.targets[self.char_bench.target]
        self.surf.blit(img, (0, y))

    def render_layers(self, vessel: Vessel,  x: int, y:int):
        """
        Method to visualize the layer information of a vessel.

        Args:
            vessel (Vessel): The vessel inputted for layer analysis.
            x (int): Position of the top left corner of this image tile
            y (int): Position of the top right corner of ths image tile
        """

        cvals = np.array([mat._color for mat in vessel._layer_mats]+[0.65])
        cnames = [mat._name for mat in vessel._layer_mats]+["air"]
        for i,name in enumerate(cnames):
            #cache the rendered text to save time
            if not name in self.misc_text:
                color = (255*cvals[i],128*cvals[i]+100,128*cvals[i]+100)
                # Drawing Rectangle
                im = self._prerender_text(" "*6+name,self.w/20)
                pygame.draw.rect(im, color, pygame.Rect(self.w/50, self.w/240, self.w/40, self.w/40))
                self.misc_text[name] = im

            self.surf.blit(self.misc_text[name], (x+self.w*2/3, y+i*20+self.w/12))


        im = np.zeros([1,100,3],dtype=np.uint8)

        im[:] = vessel.get_layers()[None,::-1,None]*255
        im[:,:,1:] = im[:,:,1:]/2
        im[:,:,1:]+=100

        surf = pygame.surfarray.make_surface(im)
        surf = pygame.transform.scale(surf,(self.w/2,self.w*5/6))
        self.surf.blit(surf, (x+self.w/6, y+self.w/12))

        bbox = [(self.w/6+x,y+self.w/12),(self.w/6+x,y+self.w*11/12),(self.w*2/3+x,y+self.w*11/12),(self.w*2/3+x,+self.w/12)]

        pygame.draw.lines(self.surf, points=bbox, closed=False, color=(0, 0, 0),width=int(self.w/120))


    def render_PVT(self, vessel: Vessel, x: int, y:int):
        """
        Method to create a bar graph of the pressure,volume, and temperature of a vessel

        Args:
            vessel (Vessel): Vessel we want pvt info from.
            x (int): Position of the top left corner of this image tile
            y (int): Position of the top right corner of ths image tile
        """
        if not "PVT" in self.misc_text:
            self.misc_text["PVT"] = pygame.Surface((self.w/4,self.w/4))
            self.misc_text["PVT"].fill((255, 255, 255))
            self.misc_text["PVT"].blit(self._prerender_text(" Temperature",self.w/20), (0,self.w/36))
            self.misc_text["PVT"].blit(self._prerender_text(" "*11+"Volume",self.w/20), (0,self.w*4/36))
            self.misc_text["PVT"].blit(self._prerender_text(" "*8+"Pressure",self.w/20), (0,self.w*7/36   ))
        t,v,p = self.char_bench.encode_PVT(vessel)
        pygame.draw.rect(self.surf, (255,0,0), pygame.Rect(x+self.w/4, y,             self.w*t*0.75, self.w/12))
        pygame.draw.rect(self.surf, (0,0,255), pygame.Rect(x+self.w/4, y+self.w/12,   self.w*v*0.75, self.w/12))
        pygame.draw.rect(self.surf, (255,30,255), pygame.Rect(x+self.w/4, y+self.w/6, self.w*p*0.75, self.w/12))
        self.surf.blit(self.misc_text["PVT"],(x, y))

        bbox = [(self.w/4+x,y),(self.w/4+x,y+self.w/4),(self.w+x-1,y+self.w/4),(self.w+x-1,y)]

        pygame.draw.lines(self.surf, points=bbox, closed=True, color=(0, 0, 0))

__backends = dict(numba=numbaVisualizer,matplotlib=matplotVisualizer, pygame=pygameVisualizer)
try:
    import pygame
    __backend="pygame"
except:
    __backend="matplotlib"

def set_backend(backend: str):
    global __backend
    if backend in __backends:
        __backend = backend
    else:
        __backend = "numba"

def use_mpl_dark(size=2):
    global RES
    set_backend("matplotlib")
    RES = size
    matplotlib.rcParams.update(
        {'figure.facecolor': "#383838", 'axes.facecolor':'#383838',
        'axes.edgecolor':'white', 'axes.labelcolor':'white',
        'text.color':'white','xtick.color':'white',
        'ytick.color':'white','font.size': 12*RES, 'figure.figsize': (size*8,size*4)})

def use_mpl_light(size=2):
    global RES
    set_backend("matplotlib")
    RES = size
    matplotlib.rcParams.update(
        {'figure.facecolor': "white", 'axes.facecolor':'white',
        'axes.edgecolor':'black', 'axes.labelcolor':'black',
        'text.color':'black','xtick.color':'black',
        'ytick.color':'black','font.size': 12*RES, 'figure.figsize': (size*8,size*4)})

def Visualizer(char_bench):
    return __backends[__backend](char_bench)