# Local Edge Dynamics and Opinion Polarization


The proliferation of social media platforms, recommender systems, and their joint societal impacts have prompted significant interest in opinion formation and evolution within social networks. In this work, we study how local dynamics in a network can drive opinion polarization. In our paper [Local Edge Dynamics and Opinion Polarization](https://arxiv.org/abs/2111.14020), we study time evolving networks under the classic Friedkin-Johnsen opinion model. Edges are iteratively added or deleted according to simple local rules, modeling decisions based on individual preferences and network recommendations.

# Python code 

Our experimental code has been written in Python3.  We recommend using a tool to manage Python virtual environments, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  There are several required Python packages:
- [NetworkX](https://networkx.org) for network/graph packages
- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [tqdm](https://github.com/tqdm/tqdm) for CLI progress bar
- [Matplotlib](https://matplotlib.org) for creating plots 

# Files and Descriptions

1. **graphfunctions.py** ([see Section 2]()): Given a NetworkX graph and associated node opinions, we define functions which compute the Friedkin-Johnsen equilibrium, polarization, disagreement, and friend-of-friend recommendations.
2. **generatedGraphExperiments.py**: Main Python script for all experiments performed on synthetic graphs, such as Erdos-Renyi and Barabasi-Albert graph models.  Each experiment is represented by one function, and the main function of the script executes a combination of the configured experiments in sequence.
3. **realGraphExperiments.py**: Main Python script for experiments performed on real-world datasets, including friend-of-friend recommendations and confirmation bias edge removal.  The main function of the script executes the configured experiments on the selected real-world dataset.
4. **realGraphData**: This folder contains three datasets from real social networks, represented as edge lists.  They can be loaded into experiments using the Python script **realGraphExperiments.py**.  Dataset references are listed below:

## Dataset References

**Twitter (Delhi 2013) & Reddit Datasets:**

Abir De, Sourangshu Bhattacharya, Parantapa Bhattacharya, Niloy Ganguly, and Soumen Chakrabarti. 2019. Learning Linear Influence Models in Social Networks from Transient Opinion Dynamics. ACM Trans. Web 13, 3, Article 16 (November 2019), 33 pages. https://doi.org/10.1145/3343483

**Facebook Egograph Dataset:**

Julian McAuley and Jure Leskovec. 2012. Learning to discover social circles in ego networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'12). Curran Associates Inc., Red Hook, NY, USA, 539–547.

# Reproducing Results

Given a correctly configured Python environment, with all of the described dependencies installed, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for synthetic and real-world networks, respectively:

- Synthetic networks: `` python3 generatedGraphExperiments.py ``
- Real-world datasets: `` python3 realGraphExperiments.py ``


# Citation

> @misc{bhalla2021localedgedyn, 
> title={Local Edge Dynamics and Opinion Polarization},
> author={Nikita Bhalla and Adam Lechowicz and Cameron Musco},
> eprint={2111.14020},
> year={2021},
> archivePrefix={arXiv},
> primaryClass={cs.SI}}
