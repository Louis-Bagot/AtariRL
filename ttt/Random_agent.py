class Random_agent():
    """Plays randomly."""
    def __init__(self):
        pass

    def play(self, env):
        return env.random_move()
