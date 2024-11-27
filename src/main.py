import os
import gymnasium as gym

from config.settings import Settings
from training.trainer import EnhancedTrainer
from agents.q_learning_agent import QLearningAgent


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    env = gym.make('CartPole-v1', render_mode='human')
    settings = Settings()

    agent = QLearningAgent(env)

    trainer = EnhancedTrainer(env=env, agent=agent, settings=settings)

    while True:
        print("\n=== Menu do Sistema de Treinamento ===")
        print("1. Treinar novo agente")
        print("2. Carregar e visualizar agente treinado")
        print("3. Sair")

        choice = input("Escolha uma opção: ")

        if choice == '1':
            print("\nIniciando treinamento...")
            trainer.train(render=True)
        elif choice == '2':
            print("\nCarregando agente treinado...")
            trainer.play()
        elif choice == '3':
            print("\nEncerrando o programa...")
            break
        else:
            print("\nOpção inválida! Por favor, escolha 1, 2 ou 3.")

    env.close()


if __name__ == "__main__":
    main()
