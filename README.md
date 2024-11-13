## WIP

High level Jiu Jitsu is about strategy as much as technique — [Gordan Ryan used to write down how he would win a match, then execute it against some of the best competitors in the game](https://www.flograppling.com/video/6943607-gordon-ryan-writes-down-his-prediction-finishes-with-triangle-submission). This is a personal project that aims to train reinforcement learning agents on the game of jui jistu and pin them against each other. To my knowledge this hasn't been done before in Jui Jitsu and is an opportunity to test different models and representations for their effectiveness.

The statespace is based on the [GrappleMap](https://github.com/Eelis/GrappleMap) database, which is a directed graph of 800 positions and 1400 transitions.

Rules:
* The game is turn-based, with possible moves being defined by the current node of the graph and the relative position of the player. Currently, players are always in either a top or bottom position, which restricts what move a player can make
* Wins are defined by either a submission (one player's only move is to tap) or based on points at the end of a game
* Points are awarded for executing certain transitions, according to International Brazilian Jiu-Jitsu Federation (IBJJF) rules

Currently, only Q-learning is implemented, but I want to extend this to algorithms like monte-carlo tree search which place more emphasis on planning. Additionally, epsilon-greedy policies can change how often a player goes for submission wins compared to point wins. Statistically, most profesiional BJJ games are won by submissions, so in a perfect simulation agents that go for more submissions should win more 


