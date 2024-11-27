import os
import gymnasium as gym

from src.config.settings import Settings

from src.agents.q_learning_agent import QLearningAgent
from src.training.trainer import Trainer


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    while True:
        print("\n=== Menu ===")
        print("1. Treinar novo agente")
        print("2. Jogar com o melhor agente")
        print("3. Sair")

        choice = input("\nEscolha uma opção: ")

        if choice == '1':
            env = gym.make('CartPole-v1')
            settings = Settings()

            agent = QLearningAgent(env)
            trainer = Trainer(env, agent, settings)

            trainer.train()
        elif choice == '2':
            env = gym.make('CartPole-v1', render_mode='human')
            settings = Settings()

            agent = QLearningAgent(env)
            trainer = Trainer(env, agent, settings)

            trainer.play()
        elif choice == '3':
            break
        else:
            print("Opção inválida!")

    env.close()


if __name__ == "__main__":
    main()
