import tkinter as tk
from tkinter import messagebox
import random
from PIL import Image, ImageTk
from blackjack_predictor import load_model, predict_action_from
import os

class BlackjackUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Blackjack AI")
        self.master.geometry("700x500")
        self.master.configure(bg="#006400")
        self.running_count = 0
        self.init_deck()
        
        # Icon if we have one??
        try:
            self.master.iconbitmap("card_icon.ico")
        except:
            pass
            
        self.setup_ui()
        self.load_card_images()

    def setup_ui(self):
        self.title_label = tk.Label(self.master, text="Blackjack AI", bg="#006400", fg="gold", font=("Helvetica", 24, "bold"))
        self.title_label.pack(pady=10)

        self.player_frame = tk.Frame(self.master, bg="#006400")
        self.player_frame.pack(pady=10, fill=tk.X)
        
        self.dealer_frame = tk.Frame(self.master, bg="#006400")
        self.dealer_frame.pack(pady=10, fill=tk.X)

        self.player_label = tk.Label(self.player_frame, text="Player's Hand:", bg="#006400", fg="white", font=("Helvetica", 16, "bold"))
        self.player_label.pack(side=tk.LEFT, padx=20)

        self.player_cards_frame = tk.Frame(self.player_frame, bg="#006400")
        self.player_cards_frame.pack(side=tk.LEFT)
        
        self.player_score = tk.Label(self.player_frame, text="Score: 0", bg="#006400", fg="white", font=("Helvetica", 14))
        self.player_score.pack(side=tk.RIGHT, padx=20)

        self.dealer_label = tk.Label(self.dealer_frame, text="Dealer's Hand:", bg="#006400", fg="white", font=("Helvetica", 16, "bold"))
        self.dealer_label.pack(side=tk.LEFT, padx=20)

        self.dealer_cards_frame = tk.Frame(self.dealer_frame, bg="#006400")
        self.dealer_cards_frame.pack(side=tk.LEFT)
        
        self.dealer_score = tk.Label(self.dealer_frame, text="Score: ?", bg="#006400", fg="white", font=("Helvetica", 14))
        self.dealer_score.pack(side=tk.RIGHT, padx=20)

        self.ai_decision = tk.Label(self.master, text="AI Decision: ", bg="#006400", fg="gold", font=("Helvetica", 18, "bold"))
        self.ai_decision.pack(pady=20)

        self.buttons_frame = tk.Frame(self.master, bg="#006400")
        self.buttons_frame.pack(pady=10)

        self.start_button = tk.Button(self.buttons_frame, text="Start Game", command=self.start_game, 
                                     bg="gold", fg="#006400", font=("Helvetica", 14, "bold"), width=12,
                                     relief=tk.RAISED, borderwidth=3)
        self.start_button.grid(row=0, column=0, padx=10)

        self.hit_button = tk.Button(self.buttons_frame, text="Hit", command=self.hit, state=tk.DISABLED, 
                                   bg="gold", fg="#006400", font=("Helvetica", 14, "bold"), width=12,
                                   relief=tk.RAISED, borderwidth=3)
        self.hit_button.grid(row=0, column=1, padx=10)

        self.stand_button = tk.Button(self.buttons_frame, text="Stand", command=self.stand, state=tk.DISABLED, 
                                     bg="gold", fg="#006400", font=("Helvetica", 14, "bold"), width=12,
                                     relief=tk.RAISED, borderwidth=3)
        self.stand_button.grid(row=0, column=2, padx=10)
        
        self.status_bar = tk.Label(self.master, text="Welcome to Blackjack AI! Press 'Start Game' to begin.", 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#004d00", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_card_images(self):
        self.card_images = {}
        
        self.default_card = self._create_card_image("red")
        self.card_back = self._create_card_image("blue")
        
    def _create_card_image(self, color):
        card = tk.PhotoImage(width=80, height=120)
        if color == "red":
            fill_color = "#aa0000"
            text_color = "white"
        elif color == "blue":
            fill_color = "#0000aa"
            text_color = "white"
        else:
            fill_color = "white"
            text_color = "black"
            
        for x in range(80):
            for y in range(120):
                if x < 3 or x >= 77 or y < 3 or y >= 117:
                    card.put("black", (x, y))
                else:
                    card.put(fill_color, (x, y))
        
        return card

    def display_cards(self, hand, frame, show_all=True):
        for widget in frame.winfo_children():
            widget.destroy()
            
        for i, card in enumerate(hand):
            if i == 1 and not show_all:
                card_label = tk.Label(frame, image=self.card_back, bg="#006400")
                card_label.image = self.card_back
            else:
                card_label = tk.Label(frame, image=self.default_card, bg="#006400")
                card_label.image = self.default_card
                
                value_label = tk.Label(card_label, text=str(card[0]), bg="#aa0000", fg="white", font=("Helvetica", 24, "bold"))
                value_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                
            card_label.pack(side=tk.LEFT, padx=5)

    def start_game(self):
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.update_display()
        self.hit_button['state'] = tk.NORMAL
        self.stand_button['state'] = tk.NORMAL
        self.ai_decide()
        self.status_bar.config(text="Game started! Your move...")

    def init_deck(self):
        self.deck = []
        for _ in range(4):  # 4 suits
            for val in range(1, 14):
                self.deck.append(val)
        random.shuffle(self.deck)
        
    def calculate_hand_value(self, hand):
        total = sum(card[1] for card in hand)
        aces = sum(1 for card in hand if card[0] == "A")
        
        while total > 21 and aces:
            total -= 10
            aces -= 1
        
        return total

    def draw_card(self):
        if not self.deck:
            self.init_deck()

        card = self.deck.pop()

        # update running count
        if 2 <= card <= 6:
            self.running_count += 1
        elif card >= 10 or card == 1:
            self.running_count -= 1

        if card == 1:
            return ("A", 11)
        elif card == 11:
            return ("J", 10)
        elif card == 12:
            return ("Q", 10)
        elif card == 13:
            return ("K", 10)
        else:
            return (str(card), card)

    def update_display(self):
        self.display_cards(self.player_hand, self.player_cards_frame)
        self.player_score.config(text=f"Score: {self.calculate_hand_value(self.player_hand)}")
        
        self.display_cards(self.dealer_hand, self.dealer_cards_frame, show_all=False)
        self.dealer_score.config(text=f"Score: {self.dealer_hand[0][1]} + ?")

    def ai_decide(self):
        model = load_model()
        player_sum = self.calculate_hand_value(self.player_hand)
        hand_is_pair = int(len(self.player_hand) == 2 and self.player_hand[0][1] == self.player_hand[1][1])
        usable_ace = int(any(card[0] == "A" for card in self.player_hand) and player_sum <= 21)
        dealer_card = self.dealer_hand[0][1]
        current_state = [player_sum, hand_is_pair, usable_ace, dealer_card, self.running_count]
        action = predict_action_from(current_state, model)
        self.ai_decision.config(text=f"AI Recommendation: {action}")

    def hit(self):
        self.player_hand.append(self.draw_card())
        self.update_display()
        self.ai_decide()
        
        player_sum = self.calculate_hand_value(self.player_hand)
        self.status_bar.config(text=f"You drew a card. Your total is now {player_sum}.")
        
        if player_sum > 21:
            self.display_cards(self.dealer_hand, self.dealer_cards_frame, show_all=True)
            self.dealer_score.config(text=f"Score: {self.calculate_hand_value(self.dealer_hand)}")
            messagebox.showinfo("Game Over", "Player busts! AI wins.")
            self.reset_game()

    def stand(self):
        self.display_cards(self.dealer_hand, self.dealer_cards_frame, show_all=True)
        self.dealer_score.config(text=f"Score: {self.calculate_hand_value(self.dealer_hand)}")
        
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
            self.display_cards(self.dealer_hand, self.dealer_cards_frame, show_all=True)
            self.dealer_score.config(text=f"Score: {self.calculate_hand_value(self.dealer_hand)}")
            self.master.update()
            self.master.after(500)
        
        player_sum = self.calculate_hand_value(self.player_hand)
        dealer_sum = self.calculate_hand_value(self.dealer_hand)

        
        if dealer_sum > 21:
            result = "Dealer busts! Player wins!"
        elif player_sum > dealer_sum:
            result = "Player wins!"
        elif dealer_sum > player_sum:
            result = "AI wins!"
        else:
            result = "It's a tie!"
            
        messagebox.showinfo("Game Over", result)
        self.reset_game()

    def reset_game(self):
        self.player_hand = []
        self.dealer_hand = []
        
        for widget in self.player_cards_frame.winfo_children():
            widget.destroy()
        for widget in self.dealer_cards_frame.winfo_children():
            widget.destroy()
            
        self.player_score.config(text="Score: 0")
        self.dealer_score.config(text="Score: 0")
        
        self.hit_button['state'] = tk.DISABLED
        self.stand_button['state'] = tk.DISABLED
        self.ai_decision.config(text="AI Recommendation: ")
        self.status_bar.config(text="Game over. Press 'Start Game' to play again.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BlackjackUI(root)
    root.mainloop()