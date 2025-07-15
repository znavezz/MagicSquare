import tkinter as tk
from tkinter import ttk
import numpy as np
from magic_square_chromosome import MagicSquareChromosome
from most_perfect_magic_square_chromosome import MostPerfectMagicSquareChromosome
from population_management import populationManagement
from SeparatePopulationManagement import IslandPopulationManagement
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")          
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class GASimulatorGUI(tk.Tk):
    def __init__(self, N, size, state=0, most_state=False, max_gens=None):
        super().__init__()
        self.title("GA Magic Square Simulator")
        self.geometry("1400x700")

        # Simulation parameters
        self.N = N
        self.size = size
        self.sim_running = False
        self.sim_thread = None
        self.state = state
        self.most_state = most_state
        self.max_generations = max_gens
        self.best_history = [] 
        self.avg_history  = []  

        # Population manager placeholder
        self.pop_mgr = None

        # Build GUI
        self._build_menu()
        self._build_controls()
        self._build_display()
        self.update_idletasks()
        self.protocol("WM_DELETE_WINDOW", self.exit)

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
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        # top area (scrollable thumbnails)
        top_frame = ttk.Frame(body)
        top_frame.grid(row=0, column=0, sticky="nsew")

        VISIBLE_H = 450
        self.pop_canvas = tk.Canvas(top_frame, bg="white", height=VISIBLE_H)
        vscroll = ttk.Scrollbar(top_frame, orient="vertical",
                                command=self.pop_canvas.yview)
        self.pop_canvas.configure(yscrollcommand=vscroll.set)

        self.pop_frame = ttk.Frame(self.pop_canvas)
        self.pop_canvas.create_window((0, 0), window=self.pop_frame, anchor="nw")
        self.pop_frame.bind(
            "<Configure>",
            lambda e: self.pop_canvas.configure(scrollregion=self.pop_canvas.bbox("all"))
        )

        self.pop_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        top_frame.grid_rowconfigure(0, weight=1)
        top_frame.grid_columnconfigure(0, weight=1)

        def _on_mousewheel(event):
            self.pop_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.pop_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.pop_canvas.bind_all("<Button-4>", lambda e: self.pop_canvas.yview_scroll(-1, "units"))
        self.pop_canvas.bind_all("<Button-5>", lambda e: self.pop_canvas.yview_scroll(1, "units"))

        # bottom area (live plot)
        self.fig, self.ax = plt.subplots(figsize=(8, 2), dpi=100)
        self.best_line, = self.ax.plot([], [], label="Best")
        self.avg_line,  = self.ax.plot([], [], label="Average")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas_plot.draw()

        plot_widget = self.canvas_plot.get_tk_widget()
        plot_widget.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)   
        body.grid_rowconfigure(1,   weight=0)   


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

    def exit(self):
        exit(0)

    def _draw_board(self,canvas, matrix, x0, y0, size, fill_color='#f0f0f0', font_size=10):
        n = len(matrix)
        cell_size = size / n

        # Draw outer border in black
        canvas.create_rectangle(x0, y0, x0 + size, y0 + size, outline='black', width=2)

        for i in range(n):
            for j in range(n):
                x = x0 + j * cell_size
                y = y0 + i * cell_size
                canvas.create_rectangle(x, y, x + cell_size, y + cell_size, fill=fill_color, outline='gray')
                canvas.create_text(
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
        # Clear the population display area
        for w in self.pop_frame.winfo_children():
            w.destroy()
        
        # Update labels
        self.gen_label.config(text=f"Perfect Square Found! Generation: {self.pop_mgr.get_generation()}")
        self.best_label.config(text=f"Fitness: 0")

        # Create a new canvas for the winner display
        winner_canvas = tk.Canvas(self.pop_frame, bg='white', width=400, height=400)
        winner_canvas.pack(expand=True)

        # Center the large square on the winner canvas
        canvas_width = 400
        canvas_height = 400
        board_size = min(canvas_width, canvas_height) * 0.8
        x0 = (canvas_width - board_size) / 2
        y0 = (canvas_height - board_size) / 2

        self._draw_board(winner_canvas, chrom.get_square(), x0, y0, board_size, fill_color='#66ccff', font_size=25)
        
        # Update scroll region
        self.pop_canvas.configure(scrollregion=self.pop_canvas.bbox("all"))

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
        fitness_values = [c.get_fitness() for c in pop]
        self.best_history.append(pop[0].get_fitness())
        self.avg_history.append(sum(fitness_values) / len(fitness_values))

        x = range(len(self.best_history))
        self.best_line.set_data(x, self.best_history)
        self.avg_line.set_data(x, self.avg_history)

        self.ax.set_xlim(0, max(1, len(self.best_history)))
        y_max = max(self.best_history + self.avg_history)
        self.ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1)
        self.canvas_plot.draw_idle()

        self.gen_label.config(text=f"Generation: {gen}")
        self.best_label.config(text=f"Best Fitness: {best.get_fitness():.4f}")

        if best.get_fitness() == 0:
            self.sim_running = False
            self._display_winner(best)
            return
        
        if self.max_generations is not None and gen >= self.max_generations:
            self.sim_running = False
            messagebox.showinfo("Simulation finished",
                                f"Stopped after {self.max_generations} generations.\n"
                                f"Best fitness reached: {pop[0].get_fitness():.4f}")
            return

        for w in self.pop_frame.winfo_children():
            w.destroy()

        board_size = 60
        pad = 10
        min_fit, max_fit = min(fitness_values), max(fitness_values)
        self.pop_canvas.update_idletasks()           
        canvas_w = self.pop_canvas.winfo_width()
        cols = max(1, (canvas_w - pad) // (board_size + pad)) - 2
            
        for idx, chrom in enumerate(pop):
            row, col = divmod(idx, cols)

            c = tk.Canvas(self.pop_frame,
                        width=board_size, height=board_size,
                        highlightthickness=0, bg='white')

            # use grid instead of place
            c.grid(row=row, column=col, padx=pad, pady=pad)

            self._draw_board(c, chrom.get_square(), 0, 0, board_size,
                            fill_color=self.fitness_to_color(chrom.get_fitness(),
                                                            min_fit, max_fit))

        # Update scroll region and continue simulation
        self.pop_canvas.configure(scrollregion=self.pop_canvas.bbox("all"))

        # Schedule next step
        self.after(300, self._run_step)

class ConfigWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magic Square Simulation Setup")

        tk.Label(self, text="Magic Square Size (N):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Magic Type:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Evolution Strategy:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Population Size:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Max Generations:").grid(row=4, column=0, padx=10, pady=5, sticky="e")

        self.n_var = tk.IntVar(value=3)
        self.type_var = tk.StringVar(value="Regular")
        self.strategy_var = tk.StringVar(value="Regular")
        self.pop_var       = tk.IntVar(value=100)     
        self.gen_var       = tk.IntVar(value=500)     
        

        n_entry = ttk.Combobox(self, textvariable=self.n_var, values=[3, 4, 5, 8], state="readonly")
        n_entry.grid(row=0, column=1, pady=5)

        type_menu = ttk.Combobox(self, textvariable=self.type_var, values=["Regular", "Most Perfect"], state="readonly")
        type_menu.grid(row=1, column=1, pady=5)

        strat_menu = ttk.Combobox(self, textvariable=self.strategy_var, values=["Regular", "Darwinian", "Lamarckian"], state="readonly")
        strat_menu.grid(row=2, column=1, pady=5)

        ttk.Entry(self, textvariable=self.pop_var, width=10).grid(row=3, column=1, pady=5)   
        ttk.Entry(self, textvariable=self.gen_var, width=10).grid(row=4, column=1, pady=5)  

        start_button = ttk.Button(self, text="Start Simulation", command=self.start_simulation)
        start_button.grid(row=5, columnspan=2, pady=10)
        

    def start_simulation(self):
        most = (self.type_var.get() == "Most Perfect")
        N = self.n_var.get()
        strategy = self.strategy_var.get()
        pop_size  = self.pop_var.get()     
        max_gens  = self.gen_var.get() 
        state = {"Regular": 0, "Darwinian": 1, "Lamarckian": 2}[strategy]

        if most and N not in [4, 8]:
            messagebox.showerror("Invalid Configuration", "Most Perfect magic squares are only supported for size 4 or 8.")
            return
        if pop_size <= 0 or max_gens <= 0:
            messagebox.showerror("Invalid Configuration", "Population size and max generations must be positive integers.")
            return

        # Close config window
        self.destroy()

        # Launch simulation GUI
        app = GASimulatorGUI(N=N, size=pop_size, state=state, most_state=most, max_gens=max_gens)
        app.set_state(state)
        app.set_most_state(most)
        app.start_sim()
        app.mainloop()

# Start the configuration window
if __name__ == '__main__':
    ConfigWindow().mainloop()
