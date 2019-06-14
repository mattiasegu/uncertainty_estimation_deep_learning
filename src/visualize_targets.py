#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import glob
from opts import parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main(FLAGS):
    folder = FLAGS.train_dir
    experiments = glob.glob(folder + "/*")
    
    steering = []
    for exp in experiments:
        csv_file = glob.glob(exp + "/sync_steering.csv")
        steering_df = pd.read_csv(csv_file[0])
        steering+=steering_df.iloc[:, 1].tolist()



    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=steering, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Radians')
    plt.ylabel('Frequency')
    plt.title('Steering Distribution of Training Samples')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    print(maxfreq)
    plt.ylim(ymax=2000)
    plt.show()

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(FLAGS)
