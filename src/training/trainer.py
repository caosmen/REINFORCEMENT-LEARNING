import numpy as np

from collections import deque

from src.utils.visualizer import Visualizer
from src.utils.logger import setup_logger


class Trainer:
    def __init__(self, env, agent, settings):
        self.env = env
        self.agent = agent
        self.settings = settings
        self.best_reward = -np.inf
        self.visualizer = Visualizer(settings.LOG_DIR)
        self.logger = setup_logger()

    def train(self):
        scores = []
        recent_scores = deque(maxlen=100)

        for episode in range(self.settings.MAX_EPISODES):
            state, _ = self.env.reset()
            total_reward = 0

            for _ in range(self.settings.MAX_STEPS):
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            scores.append(total_reward)
            recent_scores.append(total_reward)
            avg_score = np.mean(recent_scores)

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.agent.save('checkpoints/best_agent.pkl')

            if episode % 10 == 0:
                self.logger.info(
                    f"Episódio {episode}/{self.settings.MAX_EPISODES}, "
                    f"Recompensa: {total_reward}, "
                    f"Média (100 ep): {avg_score:.2f}, "
                    f"Epsilon: {self.agent.epsilon:.4f}"
                )

        self.env.close()
        self.visualizer.plot_training_results(scores)
        self.logger.info("Treinamento concluído!")

    def play(self, episodes=5):
        self.agent.load('checkpoints/best_agent.pkl')

        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                self.env.render()
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"Episódio {episode + 1}: Recompensa Total = {total_reward}")
