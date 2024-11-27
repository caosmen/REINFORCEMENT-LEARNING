import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime


matplotlib.use('Agg')


class Visualizer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def plot_training_results(self, scores, window_size=100):
        plt.figure(figsize=(10, 6))
        plt.plot(scores, alpha=0.6, label='Pontuação por Episódio')

        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(scores)), moving_avg, 'r', label=f'Média Móvel ({window_size} episódios)')

        plt.title('Resultados do Treinamento')
        plt.xlabel('Episódio')
        plt.ylabel('Pontuação')
        plt.legend()
        plt.grid(True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.log_dir, f'training_results_{timestamp}.png'))
        plt.close()
