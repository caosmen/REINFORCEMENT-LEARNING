import os
import gymnasium as gym

from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

from src.config.settings import Settings
from src.agents.q_learning_agent import QLearningAgent
from src.training.trainer import Trainer


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def train_agent():
    clear_screen()

    print("🚀 Iniciando o treinamento do agente...\n")

    env = gym.make('CartPole-v1')
    settings = Settings()
    agent = QLearningAgent(env)
    trainer = Trainer(env, agent, settings)
    trainer.train()

    print("\n✅ Treinamento concluído!")
    input("Pressione Enter para voltar ao menu principal...")


def play_agent():
    clear_screen()

    print("🎮 Iniciando o jogo com o agente treinado...\n")

    env = gym.make('CartPole-v1', render_mode='human')
    settings = Settings()
    agent = QLearningAgent(env)
    trainer = Trainer(env, agent, settings)
    trainer.play()

    print("\n✅ Jogo concluído!")
    input("Pressione Enter para voltar ao menu principal...")


def exit_program():
    clear_screen()

    print("👋 Saindo do programa. Até logo!")

    exit(0)


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    menu = ConsoleMenu(
        "🎯 CartPole Training System",
        "Selecione uma das opções abaixo:",
        clear_screen=True,
        show_exit_option=False
    )

    train_item = FunctionItem("🧠 Treinar novo agente", train_agent)
    play_item = FunctionItem("🎮 Jogar com o melhor agente", play_agent)
    exit_item = FunctionItem("🚪 Sair", exit_program)

    menu.append_item(train_item)
    menu.append_item(play_item)
    menu.append_item(exit_item)

    menu.show()


if __name__ == "__main__":
    main()
