# Implementação de Q-Learning para Ambiente CartPole

## Visão Geral do Projeto

Este projeto implementa um agente de aprendizagem por reforço utilizando o algoritmo Q-Learning para resolver o desafio do CartPole-v1 do Gymnasium (OpenAI). O objetivo é desenvolver um agente capaz de aprender a equilibrar um pêndulo invertido sobre uma base móvel, demonstrando conceitos fundamentais de aprendizagem por reforço em um ambiente prático.

## Funcionalidades Principais

O projeto oferece uma implementação robusta com diversas características importantes:

- Implementação completa do algoritmo Q-Learning
- Sistema de discretização de estados para lidar com espaço contínuo
- Política de exploração epsilon-greedy com decaimento adaptativo
- Sistema completo de logging e visualização de resultados
- Configuração flexível via variáveis de ambiente
- Arquitetura modular e extensível
- Documentação abrangente

## Estrutura do Projeto

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── main.py
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py
    │   └── q_learning_agent.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    ├── utils/
    │   ├── __init__.py
    │   ├── logger.py
    │   └── visualizer.py
    └── training/
        ├── __init__.py
        └── trainer.py
```

## Dependências Principais

- gymnasium==1.0.0
- pygame==2.6.1
- numpy==2.1.3
- pandas==2.2.3
- matplotlib==3.9.2
- seaborn==0.13.2
- python-dotenv==1.0.1
- tqdm==4.67.1
- tensorboard==2.18.0
- black==24.10.0
- pylint==3.3.1
- pytest==8.3.3

## Instalação e Configuração

### Passo 1: Preparação do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/caosmen/REINFORCEMENT-LEARNING.git
cd REINFORCEMENT-LEARNING

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Passo 2: Instalação de Dependências

```bash
pip install -r requirements.txt
```

### Passo 3: Configuração do Ambiente

```bash
# Copiar arquivo de exemplo de configuração
cp .env.example .env

# Editar configurações conforme necessário
# Abra o arquivo .env em um editor de texto
```

## Configuração

### Variáveis de Ambiente

```ini
# Configurações do Ambiente
ENVIRONMENT_NAME=CartPole-v1
LOG_DIR=logs
CHECKPOINT_DIR=checkpoints
RANDOM_SEED=42

# Hiperparâmetros do Agente
LEARNING_RATE=0.1
DISCOUNT_FACTOR=0.95
EPSILON=1.0
EPSILON_DECAY=0.995
EPSILON_MIN=0.01

# Parâmetros de Treinamento
MAX_EPISODES=1000
MAX_STEPS=500
```

## Uso do Sistema

### Executando o Treinamento

```bash
python -m src.main
```

### Monitoramento e Resultados

- Logs de treinamento: `logs/training_[timestamp].log`
- Visualizações: `logs/training_results_[timestamp].png`
- Checkpoints: `checkpoints/model_[timestamp].pkl`

## Detalhes Técnicos

### Discretização de Estado

O sistema utiliza discretização inteligente do espaço de estados:

- Posição do carrinho: [-2.4, 2.4] → 10 divisões
- Velocidade do carrinho: [-4, 4] → 10 divisões
- Ângulo do pêndulo: [-0.21, 0.21] → 10 divisões
- Velocidade angular: [-4, 4] → 10 divisões

### Algoritmo de Aprendizagem

A implementação do Q-Learning inclui:

- Política epsilon-greedy para exploração
- Atualização dinâmica da Q-table
- Sistema de recompensas adaptativo
- Controle de convergência

## Desenvolvimento

### Adicionando Novos Agentes

1. Criar nova classe em `src/agents/`
2. Herdar de `BaseAgent`
3. Implementar métodos obrigatórios
4. Registrar no sistema de configuração

## Autores e Mantenedores

- [Bruno Lemos](https://github.com/caosmen)

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.
