
import os
from matplotlib import pyplot as plt
import numpy as np

def draw_length_distribution(sentence_lengths, fig_path):
    sentence_lengths = np.sort(sentence_lengths)
    length = len(sentence_lengths)
    plt.hist(sentence_lengths[:int(length * 0.99)], bins=100, cumulative=True, edgecolor='black', alpha=0.7, density=True, histtype='stepfilled')
    plt.title('Cumulative Distribution of Sentence Lengths')
    plt.xlabel('Sentence Length')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True)
    ratio = 0.90
    plt.axhline(y=ratio, color='red', linestyle='--', label=f'{ratio*100}% Line')
    plt.axvline(x=sentence_lengths[int(length * ratio)], color='red', linestyle='--', label=f'{ratio * 100}% Line')
    plt.savefig(fig_path)
    plt.clf()
