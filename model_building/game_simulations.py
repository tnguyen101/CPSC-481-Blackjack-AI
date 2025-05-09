from blackjack_predictor import load_model, predict_action_from
import torch
import random

# creates shuffled deck 
def create_shuffled_deck(num_decks):

    card_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11]

    return card_deck * num_decks

# draws and returns a card from the deck
def draw_card(current_deck):
    card = random.choice(current_deck)
    current_deck.remove(card)
    return card

#initializes card count
def count_cards_in(dealings):

    running_count = 0

    for card in dealings:
        if(card < 6 and card > 1):
            running_count += 1
        elif(card > 10):
            running_count -= 1

    return running_count

# determines if player hadn has an ace
def find_ace_in(player_hand):
    
    for card in player_hand:
        if(card == 1):
            return 1

    return 0

# determines if there is a pair in the players hand
def find_pair_in(player_hand):

    if len(player_hand) > 2:
        return 0

    if(player_hand[0] == player_hand[1]):
        return 1
    else:
        return 0

# calculates the players total
def calculate_hand(player_hand):

    player_total = 0

    for card in player_hand:
        player_total += card

    return player_total

# compares hand between the player and dealer
def compare_hands(player_total, dealer_total):
    if player_total > 21:
        return 0
    elif dealer_total > 21:
        return 1
    elif player_total > dealer_total:
        return 1
    elif player_total < dealer_total:
        return 0
    else:
        return -1

# dealer draws until using H17 rules
def dealer_play(dealer_hand, deck):
    while calculate_hand(dealer_hand) <= 17:
        dealer_hand.append(draw_card(deck))
    return calculate_hand(dealer_hand)

# plays a hand given the models reccomended action
def play_out_hand(player_hand, dealer_hand, action, deck_state, running_count, model):
    
    if action == "surrender":
        return 0

    elif action == "double":
        player_hand.append(draw_card(deck_state))
        player_total = calculate_hand(player_hand)
        dealer_total = dealer_play(dealer_hand, deck_state)
        outcome = compare_hands(player_total, dealer_total)
        return 2 * outcome  

    elif action == "hit":

        while calculate_hand(player_hand) < 21:
            player_hand.append(draw_card(deck_state))
            if calculate_hand(player_hand) >= 17:
                break
            
        player_total = calculate_hand(player_hand)
        dealer_total = dealer_play(dealer_hand, deck_state)
        return compare_hands(player_total, dealer_total)

    elif action == "stand":
        player_total = calculate_hand(player_hand)
        dealer_total = dealer_play(dealer_hand, deck_state)
        return compare_hands(player_total, dealer_total)

    elif action == "split":
        if player_hand[0] != player_hand[1]:
            return 0 

        hand1 = [player_hand[0], draw_card(deck_state)]
        hand2 = [player_hand[1], draw_card(deck_state)]

        dealer_copy = dealer_hand.copy()

        reward1 = play_out_hand(hand1, dealer_copy[:], "hit", deck_state)
        reward2 = play_out_hand(hand2, dealer_copy[:], "hit", deck_state)

        return max(reward1, reward2)

    else:
        return 0

# simulates a hand and plays out a game
def simulate_hand(model, current_deck):

    if len(deck_state) < 52:
        current_deck = create_shuffled_deck(num_decks=6)
        running_count = 0

    initial_dealings  = [draw_card(current_deck) for _ in range(4)]
    running_count = count_cards_in(initial_dealings)
    player_hand = [initial_dealings[0], initial_dealings[1]]
    dealer_hand = [initial_dealings[2], initial_dealings[3]]



    hand_is_pair = find_pair_in(player_hand)
    usable_ace = find_ace_in(player_hand)
    player_sum = calculate_hand(player_hand)
    dealer_card = dealer_hand[0]
    input_tensor = torch.tensor([player_sum, hand_is_pair, usable_ace, dealer_card, running_count], dtype=torch.float32)

    with torch.no_grad():
        action = predict_action_from(input_tensor, model) 

    reward = play_out_hand(player_hand, dealer_hand, action, deck_state, running_count, model)

    return reward



if __name__ == "__main__":


    model = load_model()
    total_wins = 0
    total_loss = 0
    total_push = 0

    # number of games to simulate
    num_games = 100000
    deck_state = create_shuffled_deck(num_decks=6)

    for _ in range(num_games):
        if len(deck_state) < 52:  
            deck_state = create_shuffled_deck(num_decks=6)
        reward = simulate_hand(model, deck_state)
        if(reward == 0):
            total_loss += 1
        elif (reward >= 1):
            total_wins += 1
        else:
            total_push += 1



    average_win_rate = (total_wins / num_games)
    average_loss_rate = (total_loss / num_games) 
    average_push_rate = (total_push / num_games) 

    print(f"Average win per hand: {average_win_rate}")
    print(f"Average loss per hand: {average_loss_rate}")
    print(f"Average push per hand: {average_push_rate}")



