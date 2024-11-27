import os
import numpy as np
import gymnasium as gym

from collections import deque

from agents.q_learning_agent import QLearningAgent
from utils.visualizer import Visualizer
from utils.logger import setup_logger


class Trainer:
    def __init__(self, settings):
        self.settings = settings
        self.env = gym.make(settings.ENVIRONMENT_NAME, render_mode='human')
        self.agent = QLearningAgent(self.env, settings)
        self.visualizer = Visualizer(settings.LOG_DIR)
        self.logger = setup_logger()
        self.best_reward = -np.inf
        self.scores = []
        self.recent_scores = deque(maxlen=100)

        os.makedirs('checkpoints', exist_ok=True)

    def train(self, render=False):
        for episode in range(self.settings.MAX_EPISODES):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(self.settings.MAX_STEPS):
                if render and episode % 100 == 0:
                    self.env.render()

                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                position = state[0]
                angle = state[2]
                adjusted_reward = reward - 0.5 * (abs(position) + abs(angle))

                self.agent.learn(state, action, adjusted_reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            self.scores.append(total_reward)
            self.recent_scores.append(total_reward)
            avg_score = np.mean(self.recent_scores)

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.agent.save('checkpoints/best_agent.pkl')

            if episode % 10 == 0:
                self.logger.info(
                    f"Episódio {episode}/{self.settings.MAX_EPISODES}, "
                    f"Pontuação: {total_reward}, "
                    f"Média (100 ep): {avg_score:.2f}, "
                    f"Epsilon: {self.agent.epsilon:.4f}, "
                    f"Melhor: {self.best_reward}"
                )

        self.env.close()
        self.visualizer.plot_training_results(self.scores)
        self.logger.info(f"Treinamento concluído! Melhor pontuação: {self.best_reward}")

    def play(self, episodes=5):
        try:
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

                self.logger.info(f"Demonstração - Episódio {episode + 1}: "
                                 f"Recompensa Total = {total_reward}")

        except FileNotFoundError:
            self.logger.error("Modelo treinado não encontrado. Execute o treinamento primeiro.")
        except Exception as e:
            self.logger.error(f"Erro durante a demonstração: {e}")
        finally:
            self.env.close()
