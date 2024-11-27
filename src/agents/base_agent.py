from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, settings):
        self.settings = settings

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass
