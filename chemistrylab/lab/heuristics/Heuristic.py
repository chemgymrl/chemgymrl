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