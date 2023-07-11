"""

"""

import os
import sys

import numpy as np
import chemistrylab
import datetime as dt
from typing import NamedTuple, Tuple, Callable, Optional, List
from chemistrylab.benches.general_bench import *
from chemistrylab.benches.characterization_bench import CharacterizationBench
from chemistrylab.util.Visualization import pygameVisualizer
from chemistrylab.lab.shelf import Shelf
ASSETS_PATH = os.path.dirname(__file__) + "/assets/"

class Policy():
    def __call__(self,observation):
        return self.predict(observation)
    def predict(self, observation):
        """
        Args:
        - observation: Observation of the state.
        
        Returns:
            The action to be performed.
        """
        raise NotImplementedError

class Manager():
    def __init__(self, benches: Tuple[GenBench], bench_names: Tuple[str], bench_agents: Tuple[dict]):
        self.benches=benches
        self.bench_names=bench_names
        for b in self.benches:
            b.reset()
        self.hand = []
        self.shelf = Shelf([],n_working=0)
        self.bench_agents = bench_agents
    
    def swap_vessels(self, bench_idx, vessel_idx):
        if bench_idx<0:
            bench=self
        else:
            bench = self.benches[bench_idx]
        if len(self.hand)==1:
            if vessel_idx>=len(bench.shelf):
                bench.shelf.append(self.hand.pop())
            else:
                bench.shelf[vessel_idx], self.hand[0] = self.hand[0],bench.shelf[vessel_idx]
        elif len(self.hand)==0:
            if vessel_idx<len(bench.shelf):
                self.hand.append(bench.shelf.pop(vessel_idx))


    def use_bench(self, bench, policy):
        #prep the bench
        #Check for an illegal vessel setup
        if not bench.validate_shelf():
            return -1

        o = bench.re_init()
        d = False
        while not d:
            o,r,d,*_ = bench.step(policy(o))
        
        for vessel in bench.shelf:
            print (vessel.get_material_dataframe())
            print (vessel.get_solute_dataframe())
            print(vessel._layers_settle_time)
            print(vessel._variance)
            print("-"*50)

        return 0
    
    def characterize(self, observation_list):
        self.charbench = CharacterizationBench(observation_list,[],len(self.hand))
        return self.charbench(self.hand,"")


class ManualPolicy(Policy):
    def __init__(self, env, screen = None, fps=60):

        self.fps=fps
        self.keys_to_action = dict()
        #using the same system of openai gym play
        for key_combination, action in env.get_keys_to_action().items():
            key_code = tuple(
                sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
                )
            self.keys_to_action[key_code] = action

        self.pressed=[]
        self.relevant_keys = set(a for keyset in self.keys_to_action for a in keyset)
        self.screen = screen
        self.env=env
        self.noop = self.keys_to_action.get((),env.action_space.sample()*0)
        self.clock = pygame.time.Clock()
    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed.append(event.key)
        if event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed.remove(event.key)
    def predict(self,o):
        for event in pygame.event.get():
            self.process_event(event)
        self.clock.tick(self.fps)

        if self.clock.get_rawtime()>1000:
            self.pressed=[]
        

        arr = self.env.render()
        arr_min, arr_max = np.min(arr), np.max(arr)
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        #scaler = max(self.screen.get_size()) / max(pyg_img.get_size())
        #size=np.array(pyg_img.get_size())*scaler
        pyg_img = pygame.transform.scale(pyg_img, self.screen.get_size())
        self.screen.blit(pyg_img, (0, 0))

        pygame.event.pump()
        pygame.display.flip()

        return self.keys_to_action.get(tuple(sorted(self.pressed)), self.noop)


class VisualPolicy(Policy):
    def __init__(self, env, policy, screen = None, fps=60):
        self.screen = screen
        self.env=env
        self.policy = policy
        self.clock = pygame.time.Clock()

    def predict(self, observation):
        arr = self.env.render()
        arr_min, arr_max = np.min(arr), np.max(arr)
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        #scaler = max(self.screen.get_size()) / max(pyg_img.get_size())
        #size=np.array(pyg_img.get_size())*scaler
        pyg_img = pygame.transform.scale(pyg_img, self.screen.get_size())
        self.screen.blit(pyg_img, (0, 0))
        #pygame.event.pump()
        pygame.display.flip()

        return self.policy(observation)



class Button():
    def __init__(self, width, height, color = '#66c666', hover = "#66FF66",  text=None):
        
        self.text=text
        self.font = pygame.font.SysFont('Arial', int(height*2/3))
        self.surf = pygame.Surface((width, height))
        self.surf.fill(color)
        self.hover_surf = pygame.Surface((width, height))
        self.hover_surf.fill(hover)
        self.pos=np.zeros(2)-1000
        if text is not None:
            text_render = self.font.render(text, True, (0,0,0))

            offset = (np.array([width,height])-np.array(text_render.get_size()))/2

            self.surf.blit(text_render,offset)
            self.hover_surf.blit(text_render,offset)
        self.dim = np.array([width,height])

    def show(self, screen, pos):
        screen.blit(self.surf,pos)
        self.pos = np.array(pos)
    def show_hover(self,screen, pos):
        screen.blit(self.hover_surf,pos)
        self.pos = np.array(pos)
    def check_hover(self, mousepos):
        mpos = np.array(mousepos)
        return all((mpos>=self.pos)&(mpos-self.pos<=self.dim))



class Inventory():
    flasks = []
    def __init__(self, dx, dy, shelf, boxsize = 100):
        self.dx=dx
        self.dy=dy
        self.shelf=shelf
        self.boxsize = boxsize
        self.update()
        self.hover = pygame.Surface((boxsize*1.05,boxsize*1.05)).convert_alpha()
        self.hover.fill((255,255,255,128))
        self.pos=np.zeros(2)

    def vessel_thumbnails(self,vessels):
        """Simple image representation of a vessel"""
        thumbnails=[]
        font = pygame.font.SysFont('Arial', 8)
        for v in vessels:
            idx = round(v.filled_volume()*10/v.volume)
            surf = Inventory.flasks[idx].copy()
            text = font.render(v.label, True, (0,0,0))
            offset = (surf.get_size()[0]-text.get_size()[0])/2 , surf.get_size()[1] - text.get_size()[1]

            surf.blit(text,offset)
            thumbnails.append(surf)
        return thumbnails
    
    def update(self):
        self.items = self.vessel_thumbnails(self.shelf)
        self.positions = {i:i for i in range(len(self.items))}
        self.render_inventory()

    def render_inventory(self):
        """Create a surface element representing an inventory"""
        boxsize=self.boxsize
        box = pygame.Surface((boxsize*1.05,boxsize*1.05))
        rect = pygame.Rect(boxsize/20,boxsize/20,boxsize*0.95,boxsize*0.95)
        gfxdraw.box(box,rect,(255,255,255))
        surf = pygame.Surface((self.dx*boxsize+boxsize/20, self.dy*boxsize+boxsize/20))
        for x in range(self.dx):
            for y in range(self.dy):
                surf.blit(box,(x*boxsize,y*boxsize))
        #items should be a list of surfaces
        # TODO: make sure they fit into the box
        for i,pos in self.positions.items():
            if i>=self.dx*self.dy:
                break
            item = self.items[pos]
            sx,sy=item.get_size()
            surf.blit(item,((i%self.dx)*boxsize+(boxsize-sx)/2,(i//self.dx)*boxsize+(boxsize-sy)/2))

        self.surf = surf
        return surf

    def show(self, screen, pos):
        screen.blit(self.surf,pos)
        self.pos = np.array(pos)

    def show_hover(self, screen, pos, idx = None):
        screen.blit(self.surf,pos)
        self.pos = np.array(pos)
        if idx is None:
            idx = self.check_hover(pygame.mouse.get_pos())
        if idx>=0:
            x = idx%self.dx
            y = idx//self.dx
            screen.blit(self.hover,self.pos+(x*self.boxsize,y*self.boxsize))

    def check_hover(self, mousepos):
        startpos=self.pos
        boxsize=self.boxsize
        slot = np.floor((mousepos-startpos)/boxsize)
        bds = np.array([self.dx,self.dy])
        if all((slot>=0)&(slot<bds)):
            idx = int(slot[0]) + int(slot[1])*self.dx
            return idx
        return -1




class ManagerGui():
    def __init__(self, manager: Manager):
        self.manager = manager
        self.cam = np.zeros(2)
        self.screen=None
        global pygame, gfxdraw
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

    def input(self):
        """Handle key and mouse events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            elif event.type == pygame.MOUSEWHEEL:
                self.cam += np.array((event.y,event.x))*30
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
                

                print(event)
                xy = np.array(event.pos)

                idx = self.shelf_inventory.check_hover(xy)
                if idx>=0:
                    self.manager.swap_vessels(-1, idx)
                    self.shelf_inventory.update()
                    self.hand_inventory.update()
                    return
                
                sxy = np.array(self.bench.get_size())
                for i,pos in enumerate(self.benchpos):
                    #rel = xy-pos+self.cam
                    #if all((rel>=0)&(rel<sxy)):
                    if self.bench_titles[i].check_hover(xy):
                        self.bench_idx = i
                        if i<len(self.manager.benches):
                            self.bench_buttons = [Button(40,20, text = name) for name, p in manager.bench_agents[i]]
                        else:
                            self.bench_buttons = [Button(40,20, text = a) for a in ["layers","spectra","PVT"]]
                        return
                if self.bench_idx is not None:
                    self.handle_bench_click(xy)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.bench_idx = None
                if event.key == 1073741892: #f11 key
                    if self.fullscreen:
                        self.screen = pygame.display.set_mode((1280,720),pygame.RESIZABLE)
                        self.video_size = self.screen.get_size()
                    else:
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN|pygame.RESIZABLE)
                        self.video_size = self.screen.get_size()
                    self.fullscreen=not self.fullscreen
                    self.benchpos = np.array([(x*200,self.video_size[1]-self.bench.get_size()[1]) for x in range(len(self.manager.benches)+1)])

            elif event.type == pygame.VIDEORESIZE:
                self.video_size = event.size
                self.benchpos = np.array([(x*200,self.video_size[1]-self.bench.get_size()[1]) for x in range(len(self.manager.benches)+1)])
                self.screen = pygame.display.set_mode(self.video_size,pygame.RESIZABLE)
        return False

    def display_char_bench(self):
        """Display the characteriation bench output until left click is pressed"""
        arr = pygameVisualizer(self.manager.charbench).get_rgb(self.manager.hand)
        if np.product(arr.shape)>0:
            arr_min, arr_max = np.min(arr), np.max(arr)
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
            pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            self.screen.blit(pyg_img, (0, 0))
        while np.product(arr.shape)>0:
            pygame.event.pump()
            self.clock.tick(60)
            pygame.display.flip()
            if any([(event.type == pygame.MOUSEBUTTONDOWN and event.button==1) for event in pygame.event.get()]):
                return
    def display_err_message(self,message):

        font = pygame.font.SysFont('Arial', int(self.video_size[1]/8))
        text = font.render(message, True, (255,0,0))
        self.screen.blit(text, (np.array(self.video_size) - np.array(text.get_size()))/2)

        while True:
            pygame.event.pump()
            self.clock.tick(60)
            pygame.display.flip()
            if any([(event.type == pygame.MOUSEBUTTONDOWN and event.button==1) for event in pygame.event.get()]):
                return
    def handle_bench_click(self, xy):
        """Check if mouse-clicks interact with a bench / inventory menu"""
        # Handle observation display
        if self.bench_idx == len(self.manager.benches):
            for button in self.bench_buttons:
                if button.check_hover(xy):
                    _ = self.manager.characterize([button.text])
                    self.display_char_bench()
                    return
            return
                
        idx = self.bench_inventories[self.bench_idx].check_hover(xy)
        if idx>=0:
            self.manager.swap_vessels(self.bench_idx, idx)
            self.bench_inventories[self.bench_idx].update()
            self.hand_inventory.update()

        for i,button in enumerate(self.bench_buttons):
            if button.check_hover(xy):
                name, policy = self.manager.bench_agents[self.bench_idx][i]
                code = self.manager.use_bench(self.manager.benches[self.bench_idx],policy)
                self.bench_inventories[self.bench_idx].update()
                if code<0:
                    self.display_err_message("Invalid Bench Setup")

    def render(self):

        if self.screen is None:
            self.fullscreen=False
            pygame.display.init()
            self.video_size = (1280,720)
            self.screen = pygame.display.set_mode(self.video_size,pygame.RESIZABLE)
            self.clock = pygame.time.Clock()
            self.bench = pygame.image.load(ASSETS_PATH+"drawing.svg").convert_alpha()

            self.bench_titles = []
            for i in range(len(self.manager.benches)):
                self.bench_titles.append(Button(141,29, text = self.manager.bench_names[i], color="#888888", hover = "#dddddd"))

            self.bench_titles.append(Button(141,29, text = "Characterization",color="#888888", hover="#dddddd"))

            print(self.bench.get_size())
            self.benchpos = np.array([(x*200,self.video_size[1]-self.bench.get_size()[1]) for x in range(len(self.manager.benches)+1)])
            


            for i in range(11):
                tmp = pygame.image.load(ASSETS_PATH+f"vessels/rflask_{i}.png").convert_alpha()
                Inventory.flasks.append(pygame.transform.scale(tmp,(80,70)))

            self.bench_inventories = [Inventory(5, 2, bench.shelf) for bench in self.manager.benches]
            self.hand_inventory = Inventory(1,1,self.manager.hand)
            self.shelf_inventory = Inventory(4,1,self.manager.shelf)

            self.bench_idx = None

            for plist in self.manager.bench_agents:
                for name, policy in plist:
                    if (type(policy) is ManualPolicy) or (type(policy) is VisualPolicy):
                        policy.screen=self.screen



        surf = pygame.Surface(self.video_size)
        surf.fill((255, 255, 255))
        self.screen.blit(surf,(0,0))
        self.render_benches()

        self.shelf_inventory.show_hover(self.screen, np.array(self.video_size)-(410,110))

        if self.hand_inventory.items:
            offset = np.array(self.hand_inventory.items[0].get_size())/2
            self.screen.blit(self.hand_inventory.items[0],-offset+pygame.mouse.get_pos())

        self.clock.tick(60)
        pygame.event.pump()

        pygame.display.flip()
    
    def render_benches(self):
        for i,pos in enumerate(self.benchpos):
            self.screen.blit(self.bench,pos-self.cam)
            b = self.bench_titles[i]
            if b.check_hover(np.array(pygame.mouse.get_pos())):
                b.show_hover(self.screen,pos-self.cam+(12,20))
            else:
                b.show(self.screen,pos-self.cam+(12,20))
        if self.bench_idx is not None:
            #Might change this (right now characterization bench isnt included in benches)
            if self.bench_idx<len(self.manager.benches):
                inventory = self.bench_inventories[self.bench_idx]
                inventory.show_hover(self.screen,(0,0))
            buttonpos = self.benchpos[self.bench_idx]+np.array((0,-30)-self.cam)
            for idx, button in enumerate(self.bench_buttons):
                if button.check_hover(np.array(pygame.mouse.get_pos())):
                    button.show_hover(self.screen,buttonpos+(idx*50,0))
                else:
                    button.show(self.screen,buttonpos+(idx*50,0))


if __name__ == "__main__":
    from chemistrylab.benches.distillation_bench import GeneralWurtzDistill_v2 as WDBench
    from chemistrylab.benches.distillation_bench import WurtzDistillDemo_v0 as WDDemo
    from chemistrylab.benches.reaction_bench import GeneralWurtzReact_v2 as WRBench
    from chemistrylab.benches.reaction_bench import WurtzReactDemo_v0 as WRDemo
    from chemistrylab.benches.extract_bench import GeneralWurtzExtract_v2 as WEBench
    from chemistrylab.benches.extract_bench import WurtzExtractDemo_v0 as WEDemo

    from chemistrylab.lab.heuristics.ReactionHeuristics import WurtzReactHeuristic

    import pygame

    benches = [WRDemo(),WDDemo(), WEDemo()]
    policies = [[("Manual",ManualPolicy(bench))] for bench in benches]
    policies[0]+=[("Heuristic",VisualPolicy(benches[0],WurtzReactHeuristic()))]
    manager = Manager(
        benches,
        ["Reaction","Distillation", "Extraction"],
        policies
        )

    gui = ManagerGui(manager)
    while True:
        gui.render()
        if gui.input():
            break
        
