'''
title           : app.py
description     : Implementation of schelling segregation model using Python and Streamlit. 
author          : Adil Moujahid
date_created    : 20200509
date_modified   : 20200509
version         : 0.1
usage           : streamlit run app.py
python_version  : 3.7.6
'''

import random
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Schelling:
    
    def __init__(self, size, empty_ratio, similarity_threshold, n_neighbors):
        self.size = size 
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors
        
    def populate(self):
        # Ratio of races (-1, 1) and empty houses (0)
        p = [(1-self.empty_ratio)/2, self.empty_ratio, (1-self.empty_ratio)/2]
        city_size = int(np.sqrt(self.size))**2
        self.city = np.random.choice([-1, 0, 1], size=city_size, p=p)
        self.city = np.reshape(self.city, (int(np.sqrt(city_size)), int(np.sqrt(city_size))))
    
    def run(self):
        for (row, col), value in np.ndenumerate(self.city):
            race = self.city[row, col]
            neighborhood = self.city[row:row+self.n_neighbors, col:col+self.n_neighbors]
            neighborhood_size = np.size(neighborhood)
            if neighborhood_size != 1:
                n_similar = len(np.where(neighborhood == race)[0]) - 1
                similarity_ratio = n_similar / (neighborhood_size - 1.)
                is_unhappy = (similarity_ratio < self.similarity_threshold)
                if is_unhappy:
                    empty_houses = list(zip(np.where(self.city == 0)[0], np.where(self.city == 0)[1]))
                    random_house = random.choice(empty_houses)
                    self.city[random_house] = race
                    self.city[row,col] = 0

         
#Streamlit App

st.title("Schelling's Model of Segregation")

population_size = st.sidebar.slider("Population Size", 500, 10000, 2500)
empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0., 1., .2)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0., 1., .4)
n_iterations = st.sidebar.number_input("Number of Iterations", 50)

schelling = Schelling(population_size, empty_ratio, similarity_threshold, 3)
schelling.populate()

#Plot 
plt.figure(figsize=(5, 5))
cmap = ListedColormap(['red', 'white', 'royalblue'])
plt.axis('off')
plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)
city_plot = st.pyplot(plt)

progress_bar = st.progress(0)

if st.sidebar.button('Run Simulation'):
    for i in range(n_iterations):
        schelling.run()
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)
        city_plot.pyplot(plt)
        plt.close("all")
        progress_bar.progress((i+1.)/n_iterations)




