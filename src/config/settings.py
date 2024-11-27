import os


class Settings:
    def __init__(self):
        self.ENVIRONMENT_NAME = os.getenv('ENVIRONMENT_NAME', 'CartPole-v1')
        self.LOG_DIR = os.getenv('LOG_DIR', 'logs')
        self.CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', 'checkpoints')
        self.RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.1'))
        self.DISCOUNT_FACTOR = float(os.getenv('DISCOUNT_FACTOR', '0.95'))
        self.EPSILON = float(os.getenv('EPSILON', '1.0'))
        self.EPSILON_DECAY = float(os.getenv('EPSILON_DECAY', '0.995'))
        self.EPSILON_MIN = float(os.getenv('EPSILON_MIN', '0.01'))

        self.MAX_EPISODES = int(os.getenv('MAX_EPISODES', '1000'))
        self.MAX_STEPS = int(os.getenv('MAX_STEPS', '500'))
