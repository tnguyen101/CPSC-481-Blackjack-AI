import random
import csv
import pandas as pd

# randomly chooses card
def draw_card():

  standard_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

  return random.choice(standard_deck)

# randomly decides if a hand is a pair
def decide_pair_from(player_total):

  # determines if a hand can be a pair
  if(player_total % 2 == 0):

    return random.randint(0, 1)

  return 0

# generates possible delt hand
def generate_hand():

  # generates two cards for the player
  first_card = draw_card()
  second_card = draw_card()

  usable_ace = 0

  # condition if the hand has aces
  if(first_card == 1 or second_card == 1):
    usable_ace = 1

  return first_card + second_card, usable_ace


# generates a game state
def generate_game_state():

  # generates possible delt hand
  player_total, usable_ace = generate_hand()

  # determines if a hand has a pair
  hand_is_pair = decide_pair_from(player_total)

  # draws a card for the dealer
  dealer_card = draw_card()

  # generates random card count from -8 to 8
  running_count = random.randint(-8,8)

  return [player_total, hand_is_pair, usable_ace, dealer_card, running_count]

# generates optimal actions based on the current game state
def generate_optimal_action_from(current_game_state):

  player_total, hand_is_pair, usable_ace, dealer_card, running_count = \
      current_game_state


  action = 0

  '''
  Possible Actions:
    "hit": 0,
    "stand": 1,
    "double": 2,
    "split": 3,
  '''

  # pair splitting rules
  if hand_is_pair:

    # case for double A or 8
    if usable_ace or (player_total == 16):
      action = 3

    # case for double 10
    elif (player_total == 20):

      if (running_count >= 6 and dealer_card == 4) or \
         (running_count >= 5 and dealer_card == 5) or \
         (running_count >= 6 and dealer_card == 6):
        action = 2

    # case for double 9, 7, 6, 4, 2
    elif (player_total == 18 and dealer_card in [2, 3, 4, 5, 6, 8, 9]) or \
         (player_total == 14 and dealer_card in [2, 3, 4, 5, 6]) or \
         (player_total == 12 and dealer_card in [3, 4, 5, 6]) or \
         ((player_total == 6 or player_total == 4) and dealer_card in [4, 5, 6]):
       action = 3

  # soft total rules
  elif usable_ace:

    # case for soft 20
    if (player_total == 20):
      action = 1

    # case for soft 19
    elif (player_total == 19):

      if (running_count >= 3 and dealer_card == 4) or \
         (running_count >= 1 and dealer_card == 5) or \
         (running_count < 0 and dealer_card == 6):
        action = 2
      else:
        action = 1

    # case for soft 18
    elif (player_total == 18):

      if(running_count >=2 and dealer_card <= 6):
        action = 2
      elif (dealer_card == 7 or dealer_card == 8):
        action = 1
      else:
        action = 0

    # case for soft 17
    elif (player_total == 17):
      if (running_count >= 1 and dealer_card == 2):
        action = 2
      elif (dealer_card >= 3 and dealer_card <= 6):
        action = 2
      else:
        action = 0

    # case for soft 16 and 15
    elif (player_total == 16 or player_total == 15):
      if (dealer_card >= 4 and dealer_card <= 6):
        action = 2
      else:
        action = 0

    # case for soft 14 and 13
    elif (player_total == 14 or player_total == 13):
      if (dealer_card == 5 or dealer_card == 6):
        action = 2
      else:
        action = 0



  # hard total rules
  elif not usable_ace:

    # case for hard 17 and higher
    if (player_total >= 17):
      action = 1

    # case for hard 16
    elif (player_total == 16):

      if (dealer_card in [2, 3, 4, 5, 6]):
        action = 1

      elif (dealer_card in [7, 8]):
        action = 0

      elif (dealer_card == 9 and running_count >= 4) or \
           (dealer_card == 10 and running_count >= 0) or \
           (dealer_card == 11 and running_count >= 3):
        action = 2

    # case for hard 15
    elif (player_total == 15):

      if (dealer_card in [2, 3, 4, 5, 6]):
        action = 1

      elif (dealer_card in [7, 8, 9]):
        action = 0

      elif (dealer_card == 9 and running_count >= 4) or \
           (dealer_card == 10 and running_count >= 0) or \
           (dealer_card == 11 and running_count >= 11):
        action = 2

      else:
        action = 0

    # case for hard 14 or 13
    elif (player_total == 14 or player_total == 13):

      if (dealer_card >= 2 and dealer_card <= 6):
        action = 1

      else:
        action = 0

      if (player_total == 13 and running_count <= -1):
        action = 2

    # case for hard 11
    elif (player_total == 11):
      action == 2

    # case for hard 10
    elif (player_total == 10):

      if (dealer_card >=2 and dealer_card <= 9):
        action = 2

      elif (dealer_card == 10 and running_count >= 4) or \
           (dealer_card == 11 and running_count >= 3):
        action = 2

      else:
        action = 0

    # case for hard 9
    elif (player_total == 9):

      if (dealer_card >= 3 and dealer_card <= 6):
        action = 2

      elif (dealer_card == 2 and running_count >= 1) or \
           (dealer_card == 7 and running_count >= 3):
        action = 2

      else:
        action = 0

    # case for hard 8
    elif (player_total == 8):
      if (dealer_card == 6 and running_count >= 2):
        action = 2

      else:
        action = 0

    # case for hard 7 or lower
    else:
      action = 0

  # testing case
  else:
    action = 0

  return action


def create_dataset(n = 10000, filename = 'blackjack_data.csv'):

  with open(filename, 'w', newline='') as blackjack_data:

    writer = csv.writer(blackjack_data)

    writer.writerow(['player_total', 'hand_is_pair', 'usable_ace', \
                     'dealer_card', 'runnning_count', 'action'])
    for _ in range(n):

      # generates plater_total, hand_is_pair, usable_ace, dealer_card, running count
      current_game_state = generate_game_state()

      # returns actions hit, stand, surrender, can_double, and can_split
      action = generate_optimal_action_from(current_game_state)

      # pushes data into the csv
      writer.writerow(current_game_state + [action])

if __name__ == "__main__":
    create_dataset()