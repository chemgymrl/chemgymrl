class Heuristic():
    """
    Heuristic policy for a chemgymrl bench.

    The predict function is implemented the same as in a stable baselines policy, so
    you should be able to use this heuristic policy in place of an RL algorithm.

    """

    def __init__(self,env,level=1):
        self.env=env
        self.level=level
        self.step=0

    def predict(self,observation):
        """
        Get an action from an observation based off of heurstics

        :param observation: (np.ndarray) the input observation

        :return: (np.ndarray, []) the model's action and an empty array (for baselines compatability)
        """
        raise NotImplementedError