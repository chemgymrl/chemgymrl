from chemistrylab.lab.heuristics.Heuristic import Policy
from chemistrylab.lab.manager import Manager
import numpy as np
from chemistrylab.util.Visualization import pygameVisualizer
from typing import NamedTuple, Tuple, Callable, Optional, List
import os


ASSETS_PATH = os.path.dirname(__file__) + "/assets/"

class ManualPolicy(Policy):
    def __init__(self, env, screen = None, fps=60):

        global pygame
        try:import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[classic_control]`")
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
                try:
                    self.pressed.remove(event.key)
                except:pass
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
        global pygame
        try:import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[classic_control]`")

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
        
        pygame.event.pump()
        pygame.display.flip()

        return self.policy(observation)



class Button():
    def __init__(self, width, height, color = '#66c666', hover = "#66FF66",  text=None):
        
        self.text=text
        self.font = pygame.font.SysFont('Arial', int(height*2/3))
        self.surf = pygame.Surface((width, height))
        self.color,self.hover = color,hover
        
        self.hover_surf = pygame.Surface((width, height))
        
        self.pos=np.zeros(2)-1000
        self.dim = np.array([width,height])
        if text is not None:
            self.set_text(text)
    def set_text(self, text):
        self.surf.fill(self.color)
        self.hover_surf.fill(self.hover)
        width,height = self.dim
        text_render = self.font.render(text, True, (0,0,0))
        offset = (np.array([width,height])-np.array(text_render.get_size()))/2
        self.surf.blit(text_render,offset)
        self.hover_surf.blit(text_render,offset)
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
    def __init__(self, dx, dy, shelf, boxsize = 100, name = ""):
        self.dx=dx
        self.dy=dy
        self.shelf=shelf
        self.boxsize = boxsize

        self.make_text(name, boxsize)
        self.update()
        self.hover = pygame.Surface((boxsize*1.05,boxsize*1.05)).convert_alpha()
        self.hover.fill((255,255,255,128))
        self.pos=np.zeros(2)

    def make_text(self, name, boxsize):

        self.font = pygame.font.SysFont('Arial', int(boxsize/3))
        self.text_render = self.font.render(name, True, (0,0,0))
        self.text_surf = pygame.Surface((self.dx*boxsize+boxsize/20, (0.5)*boxsize))
        self.text_surf.fill((255,255,255))
        tsize=self.text_render.get_size()
        ssize = self.text_surf.get_size()
        offset = (ssize[0]-tsize[0])/2 , (ssize[1] - tsize[1])/2
        self.text_surf.blit(self.text_render,offset)

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
    
    def get_shelf_idx(self,i):
        return len([k for k,v in self.positions.items() if k<i])

    def inplace_update(self, i, shelf_idx):
        """Rearrange the dictionary of filled slots so that no vessels move when one was added/removed"""
        self.items = self.vessel_thumbnails(self.shelf)
        if shelf_idx<0:
            self.render_inventory()
            return
        if i in self.positions:
            self.positions = {k : v - (k>i) for k,v in self.positions.items()}
            del self.positions[i]
        else:
            self.positions = {k : v + (k>i) for k,v in self.positions.items()}
            self.positions[i] = shelf_idx
        self.render_inventory()
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
        surf = pygame.Surface((self.dx*boxsize+boxsize/20, (self.dy+0.5)*boxsize+boxsize/20))

        surf.blit(self.text_surf,(0,0))
        for x in range(self.dx):
            for y in range(self.dy):
                surf.blit(box,(x*boxsize,(y+0.5)*boxsize))
        #items should be a list of surfaces
        # TODO: make sure they fit into the box
        for i,pos in self.positions.items():
            if i>=self.dx*self.dy:
                break
            item = self.items[pos]
            sx,sy=item.get_size()
            surf.blit(item,((i%self.dx)*boxsize+(boxsize-sx)/2,(i//self.dx+0.5)*boxsize+(boxsize-sy)/2))

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
            y = idx//self.dx+0.5
            screen.blit(self.hover,self.pos+(x*self.boxsize,y*self.boxsize))

    def check_hover(self, mousepos):
        startpos=self.pos + (0,self.boxsize/2)
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
                            self.bench_buttons = [Button(40,20, text = name) for name, p in self.manager.bench_agents[i]]
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
        
        inventory = self.bench_inventories[self.bench_idx]
        idx = inventory.check_hover(xy)
        if idx>=0:
            shelf_idx = inventory.get_shelf_idx(idx)
            has_in_hand = len(self.manager.hand)>0
            # swap if it's in the shelf
            if idx in inventory.positions:
                self.manager.swap_vessels(self.bench_idx, shelf_idx)
            # Insert otherwise
            elif has_in_hand:
                self.manager.insert_vessel(self.bench_idx, shelf_idx)

            if idx in inventory.positions and has_in_hand or (not idx in inventory.positions and not has_in_hand):
                shelf_idx=-1

            inventory.inplace_update(idx, shelf_idx)
            self.hand_inventory.update()

        for i,button in enumerate(self.bench_buttons):
            if button.check_hover(xy):
                name, policy = self.manager.bench_agents[self.bench_idx][i]
                code = self.manager.use_bench(self.manager.benches[self.bench_idx],policy)
                self.bench_inventories[self.bench_idx].inplace_update(0,-1)
                if code<0:
                    self.display_err_message("Invalid Bench Setup")

        if self.bench_idx< len(self.bench_targets):
            target_button = self.bench_targets[self.bench_idx]
            if target_button.check_hover(xy):
                N = len(self.manager.targets)
                target_button.idx = (target_button.idx+1)%N
                self.manager.set_target(self.bench_idx,target_button.idx)
                target_button.set_text("Target: "+self.manager.targets[target_button.idx])

    def render(self):

        if self.screen is None:
            self.fullscreen=False
            pygame.display.init()
            self.video_size = (1280,720)
            self.screen = pygame.display.set_mode(self.video_size,pygame.RESIZABLE)
            self.clock = pygame.time.Clock()
            self.bench = pygame.image.load(ASSETS_PATH+"drawing.svg").convert_alpha()

            self.bench_titles = []
            self.bench_targets = []
            for i in range(len(self.manager.benches)):
                self.bench_titles.append(Button(141,29, text = self.manager.bench_names[i], color="#888888", hover = "#dddddd"))
                self.bench_targets.append(Button(280,30, text = "Target: "+self.manager.targets[0], color="#8888FF", hover = "#ccccff"))
                self.bench_targets[i].idx = 0
                self.manager.set_target(i,0)

            self.bench_titles.append(Button(141,29, text = "Characterization",color="#888888", hover="#dddddd"))

            print(self.bench.get_size())
            self.benchpos = np.array([(x*200,self.video_size[1]-self.bench.get_size()[1]) for x in range(len(self.manager.benches)+1)])
            


            for i in range(11):
                tmp = pygame.image.load(ASSETS_PATH+f"vessels/rflask_{i}.png").convert_alpha()
                Inventory.flasks.append(pygame.transform.scale(tmp,(80,70)))

            inv_names = [n+" Bench Inventory" for n in self.manager.bench_names]
            self.bench_inventories = [Inventory(5, 2, bench.shelf,name=inv_names[i]) for i,bench in enumerate(self.manager.benches)]
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

        self.shelf_inventory.show_hover(self.screen, np.array(self.video_size)-(410,160))

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

                t = self.bench_targets[self.bench_idx]
                pos=self.benchpos[self.bench_idx]
                if t.check_hover(np.array(pygame.mouse.get_pos())):
                    t.show_hover(self.screen,(10,300))
                else:
                    t.show(self.screen,(10,300))



            buttonpos = self.benchpos[self.bench_idx]+np.array((0,-30)-self.cam)
            for idx, button in enumerate(self.bench_buttons):
                if button.check_hover(np.array(pygame.mouse.get_pos())):
                    button.show_hover(self.screen,buttonpos+(idx*50,0))
                else:
                    button.show(self.screen,buttonpos+(idx*50,0))