# This code will not run by itself. It will need to be included in odeworld_v0.py 

action_mode = 'box'

# Action and observation spaces

if action_mode == 'box':
    act_low = np.zeros(self.reaction.ni.shape[0]+2, dtype=np.float32) # 2 means change of T and V
    act_high = np.ones(self.reaction.ni.shape[0]+2, dtype=np.float32) # shape[0] means change reactant
    self.action_space = gym.spaces.Box(low=act_low, high=act_high)
    absorb = self.reaction.get_spectra()
    # 4 mean time T V P
    # ni.shape[0] mean reactant
    # absorb.shape[0] man the spectra
    obs_low = np.zeros(self.reaction.ni.shape[0]+4+absorb.shape[0], dtype=np.float32)
    obs_high = np.ones(self.reaction.ni.shape[0]+4+absorb.shape[0], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

elif action_mode == 'discrete':
    self.action_space = gym.spaces.Discrete(self.reaction.ni.shape[0]+5)
    self.button_presses = np.zeros(int(self.action_space) - 1, dtype=np.int32)
    self.max_press = 10
    self.total_presses = 0
    absorb = self.reaction.get_spectra()
    obs_low = np.zeros(np.factorial(self.reaction.ni.shape[0])+self.reaction.ni.shape[0]+6+absorb.shape[0], dtype=np.float32)
    obs_high = np.ones(np.factorial(self.reaction.ni.shape[0])+self.reaction.ni.shape[0]+6+absorb.shape[0], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

elif action_mode == 'multi_discrete':
    act_high = np.zeros(self.reaction.ni.shape[0]+3)
    act_high[0] += 3 # Increase/Decrease/No change to temperature
    act_high[1] += 3 # Increase/Decrease/No change to volume
    act_high[2:-1] += 2 # Increase/No change to reactants
    act_high[-1] += 2 # Do/Do not evolve system
    self.action_space = gym.spaces.MultiDiscrete(act_high)
    self.button_presses = np.zeros(act_high.shape[0]-1, dtype=np.int32)
    self.total_presses = 0
    self.max_press = 10
    absorb = self.reaction.get_spectra()
    obs_low = np.zeros(2*self.reaction.ni.shape[0]+6+absorb.shape[0], dtype=np.float32)
    obs_high = np.ones(2*self.reaction.ni.shape[0]+6+absorb.shape[0], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

# Update state

if action_mode == 'box':
    absorb = self.reaction.get_spectra()
    self.state = np.zeros(
                        4 + # time T V P
                        self.reaction.ni.shape[0] + # reactant
                        absorb.shape[0], # spectra
                        dtype=np.float32,
                        )
    self.state[0] = self.t / self.tmax # time
    self.state[1] = (self.T - self.Tmin) / (self.Tmax - self.Tmin) # T
    self.state[2] = (self.V - self.Vmin) / (self.Vmax - self.Vmin) # V
    self.state[3] = self.reaction.get_total_pressure(self.V, self.T) / self.Pmax # total pressure
    for i in range(self.reaction.ni.shape[0]):
        self.state[i+4] = self.reaction.n[i] / self.reaction.nmax[i]
    for i in range(absorb.shape[0]):
        self.state[i + self.reaction.ni.shape[0] + 4] = absorb[i]

elif action_mode == 'discrete':
    absorb = self.reaction.get_spectra()
    self.state = np.zeros(
                        6 + # time T V P
                        self.reaction.ni.shape[0] + # reactant
                        np.factorial(self.reaction.ni.shape[0]) + # reactant
                        absorb.shape[0], # spectra
                        dtype=np.float32,
                        )
    self.state[0] = self.t / self.tmax # time
    self.state[1] = (self.T - self.Tmin) / (self.Tmax - self.Tmin) # T
    self.state[2] = (self.V - self.Vmin) / (self.Vmax - self.Vmin) # V
    self.state[3] = self.reaction.get_total_pressure(self.V, self.T) / self.Pmax # total pressure
    for i in range(self.reaction.ni.shape[0]):
        self.state[i+4] = self.reaction.n[i] / self.reaction.nmax[i]
    for i in range(self.button_presses.shape[0]):
        self.state[i + self.reaction.ni.shape[0] + 4] = self.button_presses[i]
    for i in range(absorb.shape[0]):
        self.state[i + self.reaction.ni.shape[0] + self.button_presses.shape[0] + 4] = absorb[i]

elif action_mode == 'multi_discrete':
    absorb = self.reaction.get_spectra()
    self.state = np.zeros(
                        6 + # time T V P
                        2*self.reaction.ni.shape[0] + # reactant
                        absorb.shape[0], # spectra
                        dtype=np.float32,
                        )
    self.state[0] = self.t / self.tmax # time
    self.state[1] = (self.T - self.Tmin) / (self.Tmax - self.Tmin) # T
    self.state[2] = (self.V - self.Vmin) / (self.Vmax - self.Vmin) # V
    self.state[3] = self.reaction.get_total_pressure(self.V, self.T) / self.Pmax # total pressure
    for i in range(self.reaction.ni.shape[0]):
        self.state[i+4] = self.reaction.n[i] / self.reaction.nmax[i]
    for i in range(self.button_presses.shape[0]):
        self.state[i + self.reaction.ni.shape[0] + 4] = self.button_presses[i]
    for i in range(absorb.shape[0]):
        self.state[i + self.reaction.ni.shape[0] + self.button_presses.shape[0] + 4] = absorb[i]

# Reset

if action_mode == 'box':
    self._first_render = True
    self.done = False
    self.t = 0.0
    self.T = 1.0 * self.Ti
    self.V = 1.0 * self.Vi
    self.reaction.reset() # reinitialize the reaction class
    self.plot_data = [[0.0], [self.Ti]]
    # [0] is time, [1] is Tempurature, [2:] is each species
    for i in range(self.reaction.nmax.shape[0]):
        self.plot_data.append([self.reaction.C[i]])

    self._update_state()
    return self.state

elif action_mode == 'discrete':
    self._first_render = True
    self.done = False
    self.t = 0.0
    self.T = 1.0 * self.Ti
    self.V = 1.0 * self.Vi
    self.reaction.reset() # reinitialize the reaction class
    self.button_presses = np.zeros(int(self.action_space) - 1, dtype=np.int32)
    self.total_presses = 0
    self.plot_data = [[0.0], [self.Ti]]
    # [0] is time, [1] is Tempurature, [2:] is each species
    for i in range(self.reaction.nmax.shape[0]):
        self.plot_data.append([self.reaction.C[i]])

    self._update_state()
    return self.state

elif action_mode == 'multi_discrete':
    self._first_render = True
    self.done = False
    self.t = 0.0
    self.T = 1.0 * self.Ti
    self.V = 1.0 * self.Vi
    self.reaction.reset() # reinitialize the reaction class
    self.button_presses = np.zeros(act_high.shape[0]-1, dtype=np.int32)
    self.total_presses = 0
    self.plot_data = [[0.0], [self.Ti]]
    # [0] is time, [1] is Tempurature, [2:] is each species
    for i in range(self.reaction.nmax.shape[0]):
        self.plot_data.append([self.reaction.C[i]])

    self._update_state()
    return self.state

# Step

if action_mode == 'box':
    reward = 0
    for i in range(self.reaction.n.shape[0]):
        # Actions [1:] add reactants to system, 0.0 = 0%, 1.0 = 100%
        # If action [i] is larger than the amount of reactant i, all is added
        dC = np.min([action[i+1], self.reaction.n[i]])
        self.reaction.C[i] += dC
        self.reaction.n[i] -= dC
    for i in range(self.n_steps):
        # Action [0] changes temperature by, 0.0 = -dT K, 0.5 = 0 K, 1.0 = +dT K
        # Temperature change is linear between frames
        self.T += 2.0 * self.dT * (action[0] - 0.5) / self.n_steps
        self.T = np.min([np.max([self.T, self.Tmin]), self.Tmax])
        self.V += 2.0 * self.dV * (action[1] - 0.5) / self.n_steps
        self.V = np.min([np.max([self.V, self.Vmin]), self.Vmax])
        d_reward = self.reaction.update(self.T, self.V, self.dt)
        self.t += self.dt # Increase time by time step
        reward += d_reward # Reward is change in concentration of C
        # Record data for render mode
        self.plot_data[0].append(self.t)
        self.plot_data[1].append(self.T)
        for i in range(self.reaction.C.shape[0]):
            self.plot_data[i+2].append(self.reaction.C[i])

    self._update_state()

    return self.state, reward, self.done, {}

elif action_mode == 'discrete':
    self.button_presses[action] = np.min([self.button_presses[action]+1, self.max_press])
    self.total_presses += 1
    if self.button_presses[-1] == 1 or self.total_presses > 50:
        for i in range(self.reaction.n.shape[0]):
            dC = np.min([self.button_presses[i+4]/self.max_press, self.reaction.n[i]])
            self.reaction.C[i] += dC
            self.reaction.n[i] -= dC
    for i in range(self.n_steps):
        # Action [0] changes temperature by, 0.0 = -dT K, 0.5 = 0 K, 1.0 = +dT K
        # Temperature change is linear between frames
        self.T -= self.dT * (self.button_presses[0] / self.max_press) / self.n_steps
        self.T += self.dT * (self.button_presses[1] / self.max_press) / self.n_steps
        self.T = np.min([np.max([self.T, self.Tmin]), self.Tmax])
        self.V -= self.dV * (self.button_presses[2] / self.max_press) / self.n_steps
        self.V += self.dV * (self.button_presses[3] / self.max_press) / self.n_steps
        self.V = np.min([np.max([self.V, self.Vmin]), self.Vmax])
        d_reward = self.reaction.update(self.T, self.V, self.dt)
        self.t += self.dt # Increase time by time step
        reward += d_reward # Reward is change in concentration of C
        # Record data for render mode
        self.plot_data[0].append(self.t)
        self.plot_data[1].append(self.T)
        for i in range(self.reaction.C.shape[0]):
            self.plot_data[i+2].append(self.reaction.C[i])

    self._update_state()

    return self.state, reward, self.done, {}

elif action_mode == 'multi_discrete':
    self.button_presses[0] += action[0] - 1
    self.button_presses[1] += action[1] - 1
    for i in range(self.reaction.n.shape[0]):
        self.button_presses[i+2] += action[i+2]
    self.button_presses[-1] += action[-1]
    for i in range(self.button_presses.shape[0]):
        self.button_presses[i] = np.max([np.min([self.button_presses[i], self.max_press]), -1.0*self.max_press])
    self.total_presses += 1
    if self.button_presses[-1] == 1 or self.total_presses > 50:
        for i in range(self.reaction.n.shape[0]):
            dC = np.min([self.button_presses[i+2]/self.max_press, self.reaction.n[i]])
            self.reaction.C[i] += dC
            self.reaction.n[i] -= dC
    for i in range(self.n_steps):
        # Action [0] changes temperature by, 0.0 = -dT K, 0.5 = 0 K, 1.0 = +dT K
        # Temperature change is linear between frames
        self.T += self.dT * (self.button_presses[0] / self.max_press) / self.n_steps
        self.T = np.min([np.max([self.T, self.Tmin]), self.Tmax])
        self.V += self.dV * (self.button_presses[1] / self.max_press) / self.n_steps
        self.V = np.min([np.max([self.V, self.Vmin]), self.Vmax])
        d_reward = self.reaction.update(self.T, self.V, self.dt)
        self.t += self.dt # Increase time by time step
        reward += d_reward # Reward is change in concentration of C
        # Record data for render mode
        self.plot_data[0].append(self.t)
        self.plot_data[1].append(self.T)
        for i in range(self.reaction.C.shape[0]):
            self.plot_data[i+2].append(self.reaction.C[i])

    self._update_state()

    return self.state, reward, self.done, {}

