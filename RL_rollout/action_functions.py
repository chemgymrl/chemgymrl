from SVG_Objects import *


def trace_open(x, y, points, seq, size, stroke="black", sw=5):
    """
    Generates an SVG path element for an open contour defined by a sequence of points and path commands.

    Args:
    - x (float): The x-coordinate of the top-left corner of the SVG canvas.
    - y (float): The y-coordinate of the top-left corner of the SVG canvas.
    - points (list): A list of (x, y) coordinate pairs defining the contour. 
        For the `A` command, this list should contain alternative parameters: `rx`, `ry`, `x-axis-rotation`,
        `large-arc-flag`, `sweep-flag`, `x`, and `y`.
    - seq (str): A string of path commands specifying the type of each contour segment. Use commas (,) instead of commands
        if a command requres additional x,y pairs. (ex: a Bezier cubic should be C,,)
    - size (float): The scaling factor for the coordinates.
    - stroke (str): The stroke color for the contour (default is "black").
    - sw (float): The stroke width for the contour (default is 5).

    Returns:
        A string representing an SVG path element for the open contour.
    """
    
    
    
    contour = ""
    for i,p in enumerate(points[1:]):
        if seq[i].upper()=="A":
            contour+=f"{seq[i]} {p[0]*size} {p[1]*size} {p[2]} {int(p[3])} {int(p[4])} {p[5]*size} {p[6]*size} "
        else:
            contour+=f"{seq[i]} "+" ".join([str(a*size) for a in p])+" "

    return f"""
        <g transform="translate({x}, {y})">
          <path d="M {points[0][0]*size},{points[0][1]*size} {contour}
            " fill="none" stroke="{stroke}" stroke-width="{sw}"/>
        </g>"""


def trace_closed(x,y,points,seq,size,fill="url(#grad)",stroke="none",sw=5,op=1,theta=0):
    
    """
    Generates an SVG path element for a closed contour defined by a sequence of points and path commands.

    Args:
    - x (float): The x-coordinate of the top-left corner of the SVG canvas.
    - y (float): The y-coordinate of the top-left corner of the SVG canvas.
    - points (list): A list of (x, y) coordinate pairs defining the contour. 
        For the `A` command, this list should contain alternative parameters: `rx`, `ry`, `x-axis-rotation`,
        `large-arc-flag`, `sweep-flag`, `x`, and `y`.
    - seq (str): A string of path commands specifying the type of each contour segment. Use commas (,) instead of commands
        if a command requres additional x,y pairs. (ex: a Bezier cubic should be C,,)
    - size (float): The scaling factor for the coordinates.
    - fill (str): The fill color for the contour (default is "url(#grad)").
    - stroke (str): The stroke color for the contour (default is "none").
    - sw (float): The stroke width for the contour (default is 5).
    - op (float): The opacity of the contour (default is 1).
    - theta (float): The angle of rotation for the contour (default is 0).
    Returns:
        A string representing an SVG path element for the open contour.
    """
    
    
    contour = ""
    for i,p in enumerate(points[1:]):
        if seq[i].upper()=="A":
            contour+=f"{seq[i]} {p[0]*size} {p[1]*size} {p[2]} {int(p[3])} {int(p[4])} {p[5]*size} {p[6]*size} "
        else:
            contour+=f"{seq[i]} "+" ".join([str(a*size) for a in p])+" "

    return f"""
        <g transform="translate({x}, {y}) rotate({theta})">
          <path d="M {points[0][0]*size},{points[0][1]*size} {contour}
               Z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>
        </g>"""

################################# SPECIAL PATH TRACING USING A CLIP PATH#########################

def fill_vessel(x, y, amount, points, seq, size, fill, sw=5, op=0.2, bbox=[0,0,1,1]):
    """
    Generates SVG code for filling a vessel represented by a polygonal shape with a liquid-like color.
    Note: This can only fill vessels in an upright position, hence why there is no theta variable as input
    
    Args:
    - x: x-position to place the vessel
    - y: y position to place the vessel
    - amount (float): number in [0,1] representing the percentage of the vessel's height that should be filled with the
        liquid-like color.
    - points: list of 2-tuples representing the vertices of the polygonal shape defining the vessel.
    - seq (str): A string of path commands specifying the type of each contour segment. Use commas (,) instead of commands
        if a command requres additional x,y pairs. (ex: a Bezier cubic should be C,,)
    - size (float): The scaling factor for the SVG image (larger values produce bigger images).
    - fill: string representing the vessel glass coloring 
        (in any SVG format, e.g., "#a2d0fa", "blue", "rgb(100,200,50)", etc.).
    - sw: float or int representing the width of the stroke (in SVG units) used to draw the outline of the vessel's shape.
    - op: opacity of the glass coloring (0 means fully transparent, 1 means fully opaque).
    - bbox: list of 4 floats or ints representing the coordinates of the bounding box for the vessel.
    
    Returns:
    - A string containing the SVG code for the filled vessel image.
    """

    code=np.random.randint(1000000)
    path = trace_closed(x,y,points,seq,size,fill="none",stroke="black",sw=sw)
    svg_code=path.split("<path")[0]
    
    svg_code+=f"""
    <defs>
        <clipPath id="gen-clip{code}">
            <path {path.split("<path")[1].split("fill")[0]} />
        </clipPath>
      </defs>
    """
    svg_code+=f"""
    <rect x="{bbox[0]*size}" y="{(bbox[1]+(1-amount)*bbox[3])*size}" 
    width="{bbox[2]*size}" height="{amount*bbox[3]*size}" fill="#a2d0fa" stroke="none" clip-path="url(#gen-clip{code})"/>
    """
    svg_code+=f"""
    <rect x="{bbox[0]*size}" y="{bbox[1]*size}" width="{bbox[2]*size}" height="{bbox[3]*size}" 
    fill="{fill}" opacity="{op}" stroke="none" clip-path="url(#gen-clip{code})"/>
    """
    svg_code+=path.split('rotate(0)">')[1]
    return svg_code

###################################DISTILLATION ACTIONS#########################################


DOC="""

    <DESCRIPTION>

    Inputs:
    - amount (float): Number in [0,1] describing how much of the action was performed

    Returns:
    - svg_code (str): a String containing SVG code to create the image associated with the action and amount.


"""



def DISTILL_0A(amount):
    svg_code="""
    <defs>
        <radialGradient id="grad" cx="50%" cy="50%" r="70%" fx="50%" fy="80%">
          <stop offset="0%" style="stop-color:yellow;stop-opacity:1" />
          <stop offset="40%" style="stop-color:orange;stop-opacity:1" />
          <stop offset="100%" style="stop-color:red;stop-opacity:1" />
        </radialGradient>
      </defs>
    """
    svg_code += trace_closed(54+amount*6,190-30*amount,FLAME,FLAME_INSTRUCT,50*amount)
    svg_code += trace_closed(54,62,ROUND_FLASK,ROUND_FLASK_INSTRUCT,50,fill="#deafaf",stroke="black")
    svg_code +="""
    <path d="M 5 195 L 15 115 L 90 115 L 100 195 M 35 195 L 70 195" fill="none" stroke="black" stroke-width="5"/>
    """
    
    svg_code+=trace_closed(150,145,ERLENMEYER,ERLENMEYER_INSTRUCT,50,fill="blue",stroke="black",op=0.2)
    svg_code+=trace_closed(150,145,ERLENMEYER,ERLENMEYER_INSTRUCT,50,fill="none",stroke="black")
    
    svg_code +="""
    <path d="M 60 40 L 154 118 M 60 45 L 147 118" fill="none" stroke="black" stroke-width="2"/>
    """
    
    return svg_code

def DISTILL_0B(amount):
    
    count=int((amount-0.5)*10+1.5)
    svg_code = trace_closed(90,115,ROUND_FLASK,ROUND_FLASK_INSTRUCT,75,fill="#deafaf",stroke="black")
    
    svg_code+=trace_closed(-22,17,B2_WATER,B2_WATER_INSTRUCT,225,fill="blue",op=0.5)
    svg_code+=trace_closed(-22,17,B2,B2_INSTRUCT,225,fill="none",stroke="black")
    
    allpos=[]
    for i in range(count):
        posx=48+85*np.random.random()
        posy=140+50*np.random.random()
        pos=np.array([posx,posy])
        if allpos:
            distance=min([((pos-p)**2).sum() for p in allpos])
        for x in range(1000):
            if not allpos or distance>400:
                break
            pos2=np.array([48+85*np.random.random(),140+50*np.random.random()])
            distance2 = min([((pos2-p)**2).sum() for p in allpos])
            if distance2>distance:
                distance,pos=distance2,pos2
        
        allpos+=[pos]
        theta=np.random.random()*360
        svg_code+=trace_closed(pos[0],pos[1],CUBE,CUBE_INSTRUCT,25,fill="#a2bafa",stroke="black",sw=2,op=0.4,theta=theta)
        svg_code+=trace_closed(pos[0],pos[1],CUBE_INNER,CUBE_INNER_INSTRUCT,25,fill="black",op=0.4,theta=theta)
        #svg_code+="</q>"
    return svg_code

DISTILL_0 = lambda amount: DISTILL_0A(amount) if amount>=0.5 else DISTILL_0B(1-amount)

def DISTILL_1(amount):
    theta=45+amount*90
    dx = -np.sin(theta/180*np.pi)*20
    svg_code = trace_closed(105+dx,80,ROUND_FLASK,ROUND_FLASK_INSTRUCT,50,fill="#deafaf",stroke="black",theta=theta)
    svg_code+=fill_vessel(115,150,amount,ERLENMEYER,ERLENMEYER_INSTRUCT,50,fill="blue",bbox=[-0.5,-0.45,1,1.4])    
    return svg_code


def DISTILL_2(amount):
    theta=45+amount*90
    dx = -np.sin(theta/180*np.pi)*20
    svg_code = trace_closed(105+dx,80,ERLENMEYER,ERLENMEYER_INSTRUCT,50,fill="blue",op=0.2,stroke="black",theta=theta)
    svg_code += trace_closed(105+dx,80,ERLENMEYER,ERLENMEYER_INSTRUCT,50,fill="none",stroke="black",theta=theta)
    svg_code+=fill_vessel(115,150,amount,ERLENMEYER,ERLENMEYER_INSTRUCT,50,fill="green",bbox=[-0.5,-0.45,1,1.4])    
    return svg_code
    
    

def DISTILL_3(amount):
    func=lambda x: x*0.75 if x < 0.8 else x*2-1
    svg_code = create_curved_hourglass_svg(90,120,95,120,1,func(amount),4)
    return svg_code


DISTILL_4 = lambda x: stop(90,130,80)

DISTILL_ACTIONS = [DISTILL_0,DISTILL_1,DISTILL_2,DISTILL_3,DISTILL_4]

DISTILL_NAMES = ['Heat / Cool', 'Pour 0->1', 'Pour 1->2', 'Wait', 'End Experiment' ]


for i,D in enumerate(DISTILL_ACTIONS):
    D.__doc__ = DOC.split("<DESCRIPTION>")[0]+DISTILL_NAMES[i]+DOC.split("<DESCRIPTION>")[1]
############################################EXTRACTION BENCH ACTIONS##########################################


def EXTRACT_0(amount):
    
    svg_code=fill_vessel(30,115,amount,B2,B2_INSTRUCT,120,fill="#71ab9c",bbox=[0,0.15,1,0.73])
    svg_code+= fill_vessel(38,20,1-amount,EV_0,EV_0_INSTRUCT,100,fill="none",bbox=[0.2,0.05,0.5,0.61])    
    svg_code += trace_closed(38,20,EV_1,"L"*20,100,fill="#365470",stroke="black",sw=2)
    svg_code+="""
    <path d="M 72 50 L 30 50 30 225 150 225" fill="none" stroke="black" stroke-width="10"/>
    """
    return svg_code

def EXTRACT_1(amount):

    count=int((amount)*10+1.5)+1
    svg_code=f"""
    
    <defs>
    <filter id="blur-filter">
      <feGaussianBlur in="SourceGraphic" stdDeviation="{count/5}" />
    </filter>
      </defs>
      <g filter="url(#blur-filter)">
    
    """
    
    y0=60 - 5*count/2
    
    op=np.clip(1.4/count**0.5,0,1)
    
    # make a list of heights and rearrange so that the middle heights are at the end (will be rendered on top)
    h0=np.arange(count)
    heights=h0*1.0
    s=heights[::2].shape[0]
    heights[::2]=h0[:s]
    s=heights[1::2].shape[0]
    heights[1::2]=h0[count-1:count-s-1:-1]
    
    #add in vessel (count times) with some gaussian blur for motion visualization
    for c in heights:
        svg_code += trace_closed(-10,y0+5*c,EV_0,EV_0_INSTRUCT,200,fill="#a2d0fa",stroke="black",op=op)
    
    svg_code+="</g>"
    
    #add in motion lines
    svg_code+=f"""
    <path d="M 65 {60-count*2.5} Q 90 {40-count*2.5} 115 {60-count*2.5}
    M 55 {55-count*2.5} Q 90 {30-count*2.5} 125 {55-count*2.5}
    M 65 {190+count*2.5} Q 90 {210+count*2.5} 115 {190+count*2.5}
    M 55 {195+count*2.5} Q 90 {220+count*2.5} 125 {195+count*2.5}
    " fill="none" stroke="black" stroke-width="3"/>
    """
    
    return svg_code



def EXTRACT_2_3(amount,color="none"):

    theta=45+amount*45
    
    dx = np.sin(theta/360*np.pi-0.1)*180
    dy=theta/2
    svg_code = fill_vessel(60,135,amount,EV_0,EV_0_INSTRUCT,100,fill="none",bbox=[0.2,0.05,0.5,0.61])
    
    svg_code +=trace_closed(0+dx,-15+dy,B2,B2_INSTRUCT,120,fill=color,stroke="black",sw=0,op=0.2,theta=theta)
    svg_code +=trace_closed(0+dx,-15+dy,B2,B2_INSTRUCT,120,fill="none",stroke="black",sw=5,theta=theta)
    
    return svg_code

EXTRACT_2 = lambda x: EXTRACT_2_3(x,"#703636")
EXTRACT_3 = lambda x: EXTRACT_2_3(x,"#71ab9c")


def EXTRACT_4(amount):

    theta=45+amount*90
    dx = np.sin(theta/360*np.pi-0.1)*120
    dy=theta/2
    svg_code = trace_closed(30+dx,0+dy,EV_0,EV_0_INSTRUCT,100,fill="#a2d0fa",stroke="black",theta=theta)
    svg_code +=fill_vessel(50,115,amount,B2,B2_INSTRUCT,120,fill="#703636",sw=5,op=0.2,bbox=[0,0.15,1,0.73])
    
    return svg_code


def EXTRACT_5_6(amount,is5=False):

    theta=45+amount*90
    dx = -np.sin(theta/180*np.pi)*20
    svg_code = trace_closed(95+dx,100,ROUND_FLASK,ROUND_FLASK_INSTRUCT,70,fill=["#e3cf8d","#c58de3"][is5],stroke="black",theta=theta)
    if is5:
        svg_code+=trace_open(28+dx*3+40,138-theta*0.7,C6H14,"LLLLLL",50,sw=3)
    else:    
        svg_code+=trace_open(28+dx*3+40,138-theta*0.7,ETHER_0,"LLMLL",50,sw=3)
        svg_code+=trace_open(28+dx*3+40,138-theta*0.7,ETHER_1,"LMAML",50,stroke="red",sw=3)
    
    svg_code += fill_vessel(60,135,amount,EV_0,EV_0_INSTRUCT,100,fill="none",bbox=[0.2,0.05,0.5,0.61])

    
    return svg_code

EXTRACT_5=lambda x: EXTRACT_5_6(x,True)
EXTRACT_6=lambda x: EXTRACT_5_6(x,False)

EXTRACT_7,EXTRACT_8 = DISTILL_3,DISTILL_4


EXTRACT_ACTIONS=[EXTRACT_0,EXTRACT_1,EXTRACT_2,EXTRACT_3,EXTRACT_4,EXTRACT_5,EXTRACT_6,EXTRACT_7,EXTRACT_8]

EXTRACT_NAMES = ["Drain EV to B1", "Mix EV","Pour B2 into EV","Pour B1 into EV", 
        "Pour EV into B2", "Pour S1 into EV", "Pour S2 into EV","Wait","End Experiment"]

for i,E in enumerate(EXTRACT_ACTIONS):
    E.__doc__ = DOC.split("<DESCRIPTION>")[0]+EXTRACT_NAMES[i]+DOC.split("<DESCRIPTION>")[1]

################################################## FUNCTIONS TO SHOW TRAJECTORIES ################################################

def show_actions(actions,N,all_actions):
    """
    Creates an SVG image that displays a sequence of actions along with their corresponding parameters.

    Parameters:
    - actions (str): A string containing action,param pairs. (EX: "0213" is action 0 with param 2, then action 1 with param 3)
    - N (int): A positive integer that specifies the maximum possible value of `param` for each action. 
        The value of `param` is normalized to the interval [0, 1].
    - all_actions (tuple): A tuple of functions that create SVG code for each action. Each function takes in a single parameter, 
        amount, a float in the range [0, 1], which specifies how much of the action was performed. 
        Each function returns a string containing SVG code for the image associated with the action and `amount`.

    Returns:
    - svg_code (str): A string containing SVG code that displays the sequence of actions and parameters.
    """

    svg_code=""
    for i in range(len(actions)//2):
        act=int(actions[2*i])
        param=int(actions[2*i+1])/N
        svg_code+=f'<a transform="translate({i*400}, {10})">'
        svg_code+=all_actions[act](param)
        R=130
        svg_code+=f'  <circle cx="90" cy="130" r="{R}" stroke="black" stroke-width="15" fill="none" />'
        
        if i+1<len(actions)//2:
            svg_code += f'<svg><line x1="{90+R}" y1="{130}" x2="{90+400-R}" y2="{130}" stroke="black" stroke-width="15" /></svg>'

        svg_code+="</a>"

    return svg_code



def show_actions_grouped(actions,N,all_actions):
    """
    Creates an SVG image that displays a sequence of actions along with their corresponding parameters. This is identical to show_actions
    with a single exception: neighboring actions-param pairs are grouped together if both action and parameter are identical.

    Parameters:
    - actions (str): A string containing action,param pairs. (EX: "0213" is action 0 with param 2, then action 1 with param 3)
    - N (int): A positive integer that specifies the maximum possible value of `param` for each action. 
        The value of `param` is normalized to the interval [0, 1].
    - all_actions (tuple): A tuple of functions that create SVG code for each action. Each function takes in a single parameter, 
        amount, a float in the range [0, 1], which specifies how much of the action was performed. 
        Each function returns a string containing SVG code for the image associated with the action and `amount`.

    Returns:
    - svg_code (str): A string containing SVG code that displays the sequence of actions and parameters.
    """
    
    svg_code=""
    offset=0
    prev=""
    seq=1
    
    R=130
    for i in range(len(actions)//2):
        act=int(actions[2*i])
        param=int(actions[2*i+1])/N
        
        if actions[2*i:2*i+2]!=prev:
            if i>0:
                string="x%d"%seq
                svg_code+=f"""
                <text x="{(i-offset-1)*400+84-len(string)*25}" y="{350}" font-family="Highway Gothic" fill="black" font-size="{100}">
                {string}</text>
                """
            seq=1
            svg_code+=f'<a transform="translate({(i-offset)*400}, {10})">'
            svg_code+=all_actions[act](param)
            svg_code+=f'<circle cx="90" cy="130" r="{R}" stroke="black" stroke-width="15" fill="none" />'

            svg_code+="</a>"
            
            if i>0:
                svg_code += f"""<svg><line x1="{(i-offset-1)*400+R+90}" y1="{130}" x2="{(i-offset)*400-R+90}" y2="{130}" 
                stroke="black" stroke-width="15" /></svg>"""
        
        
            prev=actions[2*i:2*i+2]
        else:
            seq+=1
            offset+=1
            
    string="x%d"%seq
    svg_code+=f"""
    <text x="{(i-offset)*400+84-len(string)*25}" y="{350}" font-family="Highway Gothic" fill="black" font-size="{100}">{string}</text>
    """
                
    return svg_code

def show_actions_mean_grouped(actions, N, all_actions, action_map, include_percents=False):
    
    """
    Returns an SVG code string containing visual representations of the given actions. Actions are grouped together by type, and
    parameters are accumulated.
    
    Inputs:
    - actions (str): A string containing action-param pairs
    - N (int): An integer representing the maximum parameter value for the actions
    - all_actions (tuple): A tuple of functions representing the visual representation of the actions
    - action_map (function): A function which maps an action and parameter to the corresponding normalized parameter 
                            and number of timesteps required for the action.
    - include_percents (bool): A boolean value indicating whether or not to include the percentage of
        of each parameter in the visual representation. Default is False.
    
    Returns:
    - svg_code (str): A string containing SVG code to create the visual representation of the actions.
    """
    
    
    svg_code=""
    offset=-1
    prev=""
    seq=0
    steps=0
    R=130

    #add in the starting set of actions
    for i in range(len(actions)//2):
        act=int(actions[2*i])
        param=(int(actions[2*i+1])+1)
        if act!=prev and i>0:
            
            if include_percents:
                string="%d"%round(seq*100)+"%"
                svg_code+=f"""
                <text x="{(i-offset-1)*400+84-len(string)*25}" y="{80}" font-family="Highway Gothic" 
                fill="black" font-size="{100}">{string}</text>
                """
            
            string="x%d"%steps
            svg_code+=f"""
            <text x="{(i-offset-1)*400+84-len(string)*25}" y="{450}" font-family="Highway Gothic" 
            fill="black" font-size="{100}">{string}</text>
            """
            
            svg_code+=f'<a transform="translate({(i-offset-1)*400}, {100})">'
            svg_code+=all_actions[prev](seq/steps)
            svg_code+=f'<circle cx="90" cy="130" r="{R}" stroke="black" stroke-width="15" fill="none" />'
            
            svg_code += f'<svg><line x1="{R+90}" y1="{130}" x2="{400-R+90}" y2="{130}" stroke="black" stroke-width="15" /></svg>'

            svg_code+="</a>"
        
            seq,steps=action_map(act,param)
            
        else:
            _seq,_steps=action_map(act,param)
            seq+=_seq
            steps+=_steps
            offset+=1
        prev=act
        
    #add in the last action 
    if include_percents:
        string="%d"%round(seq*100)+"%"
        svg_code+=f"""
            <text x="{(i-offset)*400+84-len(string)*25}" y="{80}" font-family="Highway Gothic" 
            fill="black" font-size="{100}">{string}</text>
            """
    string="x%d"%steps
    svg_code+=f"""
    <text x="{(i-offset)*400+84-len(string)*25}" y="{450}" font-family="Highway Gothic" fill="black" font-size="{100}">{string}</text>
    """
    svg_code+=f'<a transform="translate({(i-offset)*400}, {100})">'
    svg_code+=all_actions[act](seq/steps)
    svg_code+=f'<circle cx="90" cy="130" r="{R}" stroke="black" stroke-width="15" fill="none" />'
    svg_code+="</a>"
            
    return svg_code

dmap=lambda x,y:(y/10,1)

emap=lambda x,y:(y/5 if x!= 7 else 2**(y-1)/16,1)




def create_matrix(all_actions,action_names,names,vals):
    """
    Create a matrix of action functions with parameters from vals list.

    Parameters:
    - all_actions (list): A list of functions that take a parameter and return an SVG string.
    - action_names (list): A list of descriptions for each action function in all_actions.
    - names (list): A list of strings describing each parameter in vals.
    - vals (list): A list of action parameters (0-1).

    Returns:
    - svg_code: An SVG code string representing the matrix of actions and their parameters.
    """
    
    
    svg_code=""
    for j,val in enumerate(vals):
        string = names[j]
        svg_code+=f"""
        <text x="{j*300+750+84-len(string)*25}" y="{100}" font-family="Times New Roman" fill="black" font-size="{100}">{string}</text>
        """
        for i in range(len(all_actions)):
            R=130
            svg_code+=f'<a transform="translate({j*300+750}, {i*300+220})">'
            svg_code+=all_actions[i](val)
            svg_code+=f'<circle cx="90" cy="130" r="{R}" stroke="black" stroke-width="15" fill="none" />'

            svg_code+="</a>"
    for i,name in enumerate(action_names):
        svg_code+=f"""
        <text x="{10}" y="{i*300+220+150}" font-family="Times New Roman" fill="black" font-size="{100}">{name}</text>
        """
    return svg_code

