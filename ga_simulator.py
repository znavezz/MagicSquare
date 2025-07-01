import tkinter as tk
from tkinter import ttk
import numpy as np
from magic_square_chromosome import MagicSquareChromosome
from most_perfect_magic_square_chromosome import MostPerfectMagicSquareChromosome
from population_management import populationManagement
from SeparatePopulationManagement import IslandPopulationManagement
from tkinter import messagebox


class GASimulatorGUI(tk.Tk):
    def __init__(self, N, size, state=0, most_state=False):
        super().__init__()
        self.title("GA Magic Square Simulator")
        self.geometry("1400x600")

        # Simulation parameters
        self.N = N
        self.size = size
        self.sim_running = False
        self.sim_thread = None
        self.state = state
        self.most_state = most_state

        # Population manager placeholder
        self.pop_mgr = None

        # Build GUI
        self._build_menu()
        self._build_controls()
        self._build_display()
        self.update_idletasks()

    def _build_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Regular", command=lambda: self.set_state(0))
        model_menu.add_command(label="Darwinian", command=lambda: self.set_state(1))
        model_menu.add_command(label="Lamarckian", command=lambda: self.set_state(2))

        mostperf_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Mode", menu=mostperf_menu)
        mostperf_menu.add_command(label="Default", command=lambda: self.set_most_state(False))
        mostperf_menu.add_command(label="Most Perfect", command=lambda: self.set_most_state(True))

    def _build_controls(self):
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.stop_btn = ttk.Button(ctrl_frame, text="Exit", command=self.stop_sim)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.gen_label = ttk.Label(ctrl_frame, text="Generation: 0")
        self.gen_label.pack(side=tk.LEFT, padx=20)

        self.best_label = ttk.Label(ctrl_frame, text="Best Fitness: N/A")
        self.best_label.pack(side=tk.LEFT, padx=20)

    def _build_display(self):
        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def set_state(self, state):
        self.state = state

    def set_most_state(self, most):
        self.most_state = most

    def start_sim(self):
        if not self.sim_running:
            # Initialize population manager
            self.pop_mgr = IslandPopulationManagement(
                N=self.N,
                size=self.size,
                state=getattr(self, 'state', 0),
                most_state=getattr(self, 'most_state', False),
                elitism=0.2,
                mutation_rate=0.2
            )
            self.sim_running = True
            self._run_step()

    def stop_sim(self):
        self.sim_running = False
        self.destroy()
        ConfigWindow().mainloop()

    def _draw_board(self, matrix, x0, y0, size, fill_color='#f0f0f0', font_size=10):
        n = len(matrix)
        cell_size = size / n

        # Draw outer border in black
        self.canvas.create_rectangle(x0, y0, x0 + size, y0 + size, outline='black', width=2)

        for i in range(n):
            for j in range(n):
                x = x0 + j * cell_size
                y = y0 + i * cell_size
                self.canvas.create_rectangle(x, y, x + cell_size, y + cell_size, fill=fill_color, outline='gray')
                self.canvas.create_text(
                    x + cell_size / 2, y + cell_size / 2,
                    text=str(matrix[i][j]),
                    font=('Arial', font_size, 'bold')
                )
    
    def fitness_to_color(self, fitness, min_fitness, max_fitness):
        # Normalize fitness to [0, 1]
        if max_fitness == min_fitness:
            ratio = 0
        else:
            ratio = (fitness - min_fitness) / (max_fitness - min_fitness)

        # Interpolate between blue (0,0,255) and red (255,0,0)
        r = int(255 * ratio)
        g = 0
        b = int(255 * (1 - ratio))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _display_winner(self, chrom):
        self.canvas.delete("all")
        self.gen_label.config(text="Perfect Square Found!")
        self.best_label.config(text=f"Fitness: 0")
        self.gen_label.config(text=f"Perfect Square Found! Generation: {self.pop_mgr.get_generation()}")

        # Center the large square
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        board_size = min(canvas_width, canvas_height) * 0.8
        x0 = (canvas_width - board_size) / 2
        y0 = (canvas_height - board_size) / 2

        self._draw_board(chrom.get_square(), x0, y0, board_size, fill_color='#66ccff',font_size=25) 

    def _run_step(self):
        if not self.sim_running:
            return

        self.pop_mgr.evaluate_population()
        pop = sorted(
            self.pop_mgr.get_population(),
            key=lambda c: c.get_fitness()
        )
        gen = self.pop_mgr.get_generation()
        best = pop[0]  

        self.gen_label.config(text=f"Generation: {gen}")
        self.best_label.config(text=f"Best Fitness: {best.get_fitness():.4f}")

        if best.get_fitness() == 0:
            self.sim_running = False
            self._display_winner(best)
            return

        self.canvas.delete("all")
        board_size = 60
        pad = 10
        cols = 20

        fitness_values = [chrom.get_fitness() for chrom in pop]
        min_fit = min(fitness_values)
        max_fit = max(fitness_values)

        for idx, chrom in enumerate(pop):
            row = idx // cols
            col = idx % cols
            x0 = pad + col * (board_size + pad)
            y0 = pad + row * (board_size + pad)
            color = self.fitness_to_color(chrom.get_fitness(), min_fit, max_fit)
            self._draw_board(chrom.get_square(), x0, y0, board_size, fill_color=color)

        self.after(300, self._run_step)

class ConfigWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magic Square Simulation Setup")

        tk.Label(self, text="Magic Square Size (N):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Magic Type:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Evolution Strategy:").grid(row=2, column=0, padx=10, pady=5, sticky="e")

        self.n_var = tk.IntVar(value=3)
        self.type_var = tk.StringVar(value="Regular")
        self.strategy_var = tk.StringVar(value="Regular")

        n_entry = ttk.Combobox(self, textvariable=self.n_var, values=[3, 4, 5, 8], state="readonly")
        n_entry.grid(row=0, column=1, pady=5)

        type_menu = ttk.Combobox(self, textvariable=self.type_var, values=["Regular", "Most Perfect"], state="readonly")
        type_menu.grid(row=1, column=1, pady=5)

        strat_menu = ttk.Combobox(self, textvariable=self.strategy_var, values=["Regular", "Darwinian", "Lamarckian"], state="readonly")
        strat_menu.grid(row=2, column=1, pady=5)

        start_button = ttk.Button(self, text="Start Simulation", command=self.start_simulation)
        start_button.grid(row=3, columnspan=2, pady=10)

    def start_simulation(self):
        most = (self.type_var.get() == "Most Perfect")
        N = self.n_var.get()
        strategy = self.strategy_var.get()
        state = {"Regular": 0, "Darwinian": 1, "Lamarckian": 2}[strategy]

        if most and N not in [4, 8]:
            messagebox.showerror("Invalid Configuration", "Most Perfect magic squares are only supported for size 4 or 8.")
            return

        # Close config window
        self.destroy()

        # Launch simulation GUI
        app = GASimulatorGUI(N=N, size=100, state=state, most_state=most)
        app.set_state(state)
        app.set_most_state(most)
        app.start_sim()
        app.mainloop()

# Start the configuration window
if __name__ == '__main__':
    ConfigWindow().mainloop()
