import tkinter as tk

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
        self.start_button = tk.Button(self.master, text="Start Game")
        self.start_button.pack(pady=5)

        self.hit_button = tk.Button(self.master, text="Hit", state=tk.DISABLED)
        self.hit_button.pack(pady=5)

        self.stand_button = tk.Button(self.master, text="Stand", state=tk.DISABLED)
        self.stand_button.pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlackjackUI(root)
    root.mainloop()