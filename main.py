import tkinter as tk
from tkinter import messagebox
import random

class BlackjackUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Blackjack AI")
        self.master.geometry("600x400")
        self.master.configure(bg="green")

        self.setup_ui()

    def setup_ui(self):
        # Player's hand display
        self.player_label = tk.Label(self.master, text="Player's Hand:", bg="green", fg="white", font=("Arial", 14))
        self.player_label.pack(pady=10)

        self.player_cards = tk.Label(self.master, text="", bg="green", fg="white", font=("Arial", 12))
        self.player_cards.pack()

        # Dealer's hand display
        self.dealer_label = tk.Label(self.master, text="Dealer's Hand:", bg="green", fg="white", font=("Arial", 14))
        self.dealer_label.pack(pady=10)

        self.dealer_cards = tk.Label(self.master, text="", bg="green", fg="white", font=("Arial", 12))
        self.dealer_cards.pack()

        # AI decision display
        self.ai_decision = tk.Label(self.master, text="AI Decision: ", bg="green", fg="white", font=("Arial", 14, "bold"))
        self.ai_decision.pack(pady=20)

        # Control buttons
        self.start_button = tk.Button(self.master, text="Start Game", command=self.start_game)
        self.start_button.pack(pady=5)

        self.hit_button = tk.Button(self.master, text="Hit", command=self.hit, state=tk.DISABLED)
        self.hit_button.pack(pady=5)

        self.stand_button = tk.Button(self.master, text="Stand", command=self.stand, state=tk.DISABLED)
        self.stand_button.pack(pady=5)

    def start_game(self):
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.update_display()
        self.hit_button['state'] = tk.NORMAL
        self.stand_button['state'] = tk.NORMAL
        self.ai_decide()

    def draw_card(self):
        return random.randint(1, 11)

    def update_display(self):
        self.player_cards.config(text=f"Cards: {self.player_hand}")
        self.dealer_cards.config(text=f"Cards: [{self.dealer_hand[0]}, ?]")

    def ai_decide(self):
        player_sum = sum(self.player_hand)
        if player_sum < 17:
            decision = "Hit"
        else:
            decision = "Stand"
        self.ai_decision.config(text=f"AI Decision: {decision}")

    def hit(self):
        self.player_hand.append(self.draw_card())
        self.update_display()
        self.ai_decide()
        if sum(self.player_hand) > 21:
            messagebox.showinfo("Game Over", "Player busts! AI wins.")
            self.reset_game()

    def stand(self):
        while sum(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
        self.dealer_cards.config(text=f"Cards: {self.dealer_hand}")
        player_sum = sum(self.player_hand)
        dealer_sum = sum(self.dealer_hand)
        if dealer_sum > 21 or player_sum > dealer_sum:
            messagebox.showinfo("Game Over", "Player wins!")
        elif dealer_sum > player_sum:
            messagebox.showinfo("Game Over", "AI wins!")
        else:
            messagebox.showinfo("Game Over", "It's a tie!")
        self.reset_game()

    def reset_game(self):
        self.player_hand = []
        self.dealer_hand = []
        self.update_display()
        self.hit_button['state'] = tk.DISABLED
        self.stand_button['state'] = tk.DISABLED
        self.ai_decision.config(text="AI Decision: ")

if __name__ == "__main__":
    root = tk.Tk()
    app = BlackjackUI(root)
    root.mainloop()