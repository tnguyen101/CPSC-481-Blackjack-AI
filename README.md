<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url] [![Issues][issues-shield]][issues-url]

<br />
<div align="center">
<a href="https://github.com/tnguyen101/CPSC-481-Blackjack-AI">
  <img src="./img/playing_cards.svg" alt="Logo" height="60" style="filter: invert(1);">
</a>

  <h3 align="center">BlackJack AI</h3>

  <p align="center">
    Play BlackJack against an advanced AI Model!
    <br />
    <br />
    <a href="https://github.com/tnguyen101/CPSC-481-Blackjack-AI/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/tnguyen101/CPSC-481-Blackjack-AI/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## Description

&ensp;&ensp;&ensp;&ensp; 
The following program uses an model trained using Pytorch in order to predict the best move
    in a potential blackjack game using card counting. The model takes in a state vector 
    including: the current players total, whether their hand is a pair, whether they have a 
    usable ace, the dealers visible card, and the current running_count; the model will then 
    output a numerical value ranging from 0 to 4 representing whether the best move is stand, 
    hit, double, or split.

&ensp;&ensp;&ensp;&ensp; 
The model was trained using a synthetic dataset and is meant to simulate an "advanced player",
    not an ideal player, that has the ability to card count. The dataset includes 10,000 states
    of randomly generated games that were attributed an an idea action based on H17 deviation rules.
    The model learned from these game states in order to provide an ideal action given a state. Future
    implementations of this game will allow the player to select different difficutlies of a model.

&ensp;&ensp;&ensp;&ensp; 
Users will be able to use the following program to play BlackJack in the presence of a dealer and one
    or more AI players that make decision based on the model. Follow the instrucitons below to try 
    to win against our BlackJackAI.

## Getting Started

#### Start by cloning the repository
```
git clone https://github.com/tnguyen101/CPSC-481-Blackjack-AI
```

## Dependencies
None

#### Note:
This program was made using Python.
The program was designed be opened from a command line.

## Executing program

#### 1. Set your directory within the repository files
 
#### 2. Type the following in the commandline
```
python main.py
```

#### Follow the directions provided by the program

## Foler Structure

The project folder structure is organized as follows:

```
CPSC-481-BlackJack-AI/
├── img                                         # Program logo
├── model_building/                             # Files that contributed to building the model
│   ├── blackjack_dataset.csv/                  # Dataset used to train the model
│   ├── blackjack_dataset_creation.py/          # Program to create synthetic data 
│   ├── blackjack_model_creation.py/            # Program to creat the pytorch model
│   ├── game_simulations.py/                    # Simulates games and output the win, tie, and loss rate 
├── README.md                                   # Project documentation
├── blackjack_model.pth/                        # Pytorch Model
├── blackjack_predictor.py/                     # Import module to use the pytorch model
├── main.py/                                    # Images and icons used in the project

```

## Authors

Arturo Flores, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton<br>
&ensp;&ensp;&ensp;&ensp;  <br>

Travis Nguyen, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton<br>
&ensp;&ensp;&ensp;&ensp;  <br>

Johnny Nguyen, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton <br> 
&ensp;&ensp;&ensp;&ensp; <br>

Noah Yarbough, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton <br>
&ensp;&ensp;&ensp;&ensp; <br>


## Top Contributors

<a href="https://github.com/tnguyen101/CPSC-481-Blackjack-AI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tnguyen101/CPSC-481-Blackjack-AI" />
</a>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/tnguyen101/CPSC-481-Blackjack-AI.svg?style=for-the-badge
[contributors-url]: https://github.com/tnguyen101/CPSC-481-Blackjack-AI/contributors
[issues-shield]: https://img.shields.io/github/issues/tnguyen101/CPSC-481-Blackjack-AI.svg?style=for-the-badge
[issues-url]: https://github.com/tnguyen101/CPSC-481-Blackjack-AI/issues

