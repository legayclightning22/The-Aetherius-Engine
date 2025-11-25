import asyncio
import logging
import math
import random
import time
import uuid
import threading
import tkinter as tk
import colorsys
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AetheriusCore")

# --- Type Definitions ---
T = TypeVar("T")
StateVector = List[complex]

class QuantumState(Enum):
    SUPERPOSITION = auto()
    COLLAPSED = auto()
    ENTANGLED = auto()
    DECOHERENT = auto()

# --- Advanced Decorators ---

def atomic_transaction(max_retries: int = 3):
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    await asyncio.sleep(0.01)
            raise RuntimeError(f"Atomic transaction failed after {max_retries} attempts")
        return wrapper
    return decorator

def measure_execution_entropy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time
        return result
    return wrapper

# --- Metaclasses & Abstractions ---

class ComponentRegistry(ABCMeta):
    _registry: Dict[str, Type] = {}
    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)
        if hasattr(new_class, "component_id"):
            mcs._registry[name] = new_class
        return new_class

class AbstractNeuron(ABC, metaclass=ComponentRegistry):
    component_id: str = "ABSTRACT_NODE"
    def __init__(self, uid: str):
        self.uid = uid
        self.neighbors: Set['AbstractNeuron'] = set()
    
    @abstractmethod
    async def integrate_signals(self, signals: List[complex]) -> None: pass

    @abstractmethod
    def collapse_wavefunction(self) -> float: pass

# --- Core Data Structures ---

@dataclass(frozen=True)
class QubitSignal:
    source_id: str
    target_id: str
    amplitude: float
    phase: float
    timestamp: float = field(default_factory=time.time)

# --- Implementation Components ---

class HyperGraphNode(AbstractNeuron):
    component_id = "HYPER_NODE"

    def __init__(self, uid: str, max_x: int, max_y: int):
        super().__init__(uid)
        
        # Physics State (Visuals)
        self.x = random.randint(50, max_x - 50)
        self.y = random.randint(50, max_y - 50)
        self.dx = random.uniform(-0.8, 0.8)
        self.dy = random.uniform(-0.8, 0.8)
        self.energy = 0.0
        self.is_dragging = False # New flag for mouse interaction
        
        # Quantum State (Logic)
        self._state_vector: StateVector = [complex(random.random(), random.random()) for _ in range(8)]
        self.quantum_state = QuantumState.SUPERPOSITION
        self.phase_angle = random.uniform(0, 2 * math.pi)
        self.is_entangled = False

    def update_physics(self, width: int, height: int):
        # Skip physics update if user is dragging the node
        if self.is_dragging:
            return

        self.x += self.dx
        self.y += self.dy
        
        # Smooth Wall bounce
        margin = 20
        if self.x <= margin: self.dx = abs(self.dx)
        if self.x >= width - margin: self.dx = -abs(self.dx)
        if self.y <= margin: self.dy = abs(self.dy)
        if self.y >= height - margin: self.dy = -abs(self.dy)
        
        # Rotate Phase (affects color)
        self.phase_angle = (self.phase_angle + 0.05) % (2 * math.pi)
        
        # Decay energy
        self.energy *= 0.95
        if self.energy < 0.01: self.is_entangled = False

    @atomic_transaction(max_retries=5)
    async def integrate_signals(self, signals: List[complex]) -> None:
        if self.quantum_state == QuantumState.DECOHERENT: return
        self.energy = min(self.energy + 0.4, 1.0) 
        await asyncio.sleep(0.001)

    @measure_execution_entropy
    def collapse_wavefunction(self) -> float:
        probabilities = [abs(c)**2 for c in self._state_vector]
        outcome = sum(p * i for i, p in enumerate(probabilities))
        self.quantum_state = QuantumState.COLLAPSED
        self.energy = 1.0 # Flash on collapse
        return outcome

    def reset(self):
        self.quantum_state = QuantumState.SUPERPOSITION

# --- The Simulation Engine ---

class AetheriusKernel:
    def __init__(self, node_count: int, width: int, height: int):
        self.node_count = node_count
        self.width = width
        self.height = height
        self.nodes: Dict[str, HyperGraphNode] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.log_buffer = [] 
        
        # Shared Metrics
        self.metrics = {
            "entropy": 0.0,
            "coherence": 100.0,
            "born_prob": 0.0,
            "status": "INITIALIZING"
        }

    def log(self, message: str):
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(entry)
        self.log_buffer.append(entry)
        if len(self.log_buffer) > 50:
            self.log_buffer.pop(0)

    def initialize_topology(self):
        self.log("Initializing Hilbert Space Topology...")
        for i in range(self.node_count):
            uid = f"NODE_{uuid.uuid4().hex[:4].upper()}"
            self.nodes[uid] = HyperGraphNode(uid, self.width, self.height)
        self.log(f"Topology constructed: {self.node_count} Nodes.")
        self.metrics["status"] = "ONLINE"

    async def _process_signal_propagation(self):
        while self.running:
            try:
                if not self.event_queue.empty():
                    signal: QubitSignal = await self.event_queue.get()
                    if signal.target_id in self.nodes:
                        await self.nodes[signal.target_id].integrate_signals([complex(signal.amplitude, signal.phase)])
                await asyncio.sleep(0.005)
            except Exception:
                pass

    async def run_simulation_loop(self):
        self.running = True
        self.log("Engaging Aetherius Kernel...")
        workers = [asyncio.create_task(self._process_signal_propagation()) for _ in range(4)]
        
        cycle = 0
        while self.running:
            # Physics Update
            for node in self.nodes.values():
                node.update_physics(self.width, self.height)
            
            # Quantum Tunneling (Random long-distance jumps)
            if random.random() < 0.05:
                n1, n2 = random.sample(list(self.nodes.values()), 2)
                await self.event_queue.put(QubitSignal("TUNNEL", n2.uid, 1.0, 0.0))
                self.log(f"Quantum Tunneling detected: {n1.uid} -> {n2.uid}")

            # Entanglement Events
            if random.random() < 0.02:
                n1, n2 = random.sample(list(self.nodes.values()), 2)
                n1.is_entangled = True
                n2.is_entangled = True
                n1.energy = 1.0
                n2.energy = 1.0

            # Random Stimulus
            if random.random() < 0.3:
                target = random.choice(list(self.nodes.values()))
                await self.event_queue.put(QubitSignal("EXT", target.uid, 0.5, 0.0))

            # Metrics Calculation
            if cycle % 10 == 0:
                self._update_metrics(cycle)
            
            cycle += 1
            await asyncio.sleep(0.016)

    def _update_metrics(self, cycle_id: int):
        active = sum(1 for n in self.nodes.values() if n.energy > 0.1)
        entropy = active / self.node_count
        
        self.metrics["entropy"] = entropy
        self.metrics["coherence"] = (1.0 - entropy) * 100
        self.metrics["born_prob"] = random.random()
        
        if entropy > 0.8: self.metrics["status"] = "CRITICAL FLUX"
        elif entropy > 0.4: self.metrics["status"] = "OPTIMAL"
        else: self.metrics["status"] = "STABLE"

        if cycle_id % 100 == 0:
             self.log(f"System Entropy: {entropy:.4f} J/K | Status: {self.metrics['status']}")

# --- Graphical User Interface ---

class Visualizer(tk.Tk):
    def __init__(self, kernel: AetheriusKernel):
        super().__init__()
        self.kernel = kernel
        self.title("Aetherius Engine :: Quantum Visualizer Pro")
        self.geometry(f"{kernel.width + 320}x{kernel.height}")
        self.configure(bg="#050505")
        
        # Layout
        self.canvas = tk.Canvas(self, width=kernel.width, height=kernel.height, bg="#000000", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        
        self.sidebar = tk.Frame(self, width=320, bg="#0f0f12", padx=10, pady=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.sidebar.pack_propagate(False)
        
        # 1. Header
        tk.Label(self.sidebar, text=":: AETHERIUS KERNEL ::", fg="#00ff9d", bg="#0f0f12", font=("Courier", 14, "bold")).pack(pady=(0, 20))
        
        # 2. Metrics Panel
        self.metrics_vars = {
            "status": tk.StringVar(value="OFFLINE"),
            "entropy": tk.StringVar(value="0.0000"),
            "coherence": tk.StringVar(value="100.0%"),
            "born": tk.StringVar(value="0.0000")
        }
        
        self._create_metric_row("SYSTEM STATUS", self.metrics_vars["status"], "#00bcd4")
        self._create_metric_row("ENTROPY (J/K)", self.metrics_vars["entropy"], "#ff5252")
        self._create_metric_row("COHERENCE", self.metrics_vars["coherence"], "#00ff9d")
        self._create_metric_row("BORN PROB", self.metrics_vars["born"], "#a0a0a0")
        
        tk.Frame(self.sidebar, height=2, bg="#333").pack(fill=tk.X, pady=20)

        # 3. Log Box
        tk.Label(self.sidebar, text=":: EVENT LOGS ::", fg="#a0a0a0", bg="#0f0f12", font=("Courier", 10)).pack(anchor="w")
        self.log_list = tk.Listbox(self.sidebar, bg="#0f0f12", fg="#777", font=("Consolas", 8), borderwidth=0, highlightthickness=0)
        self.log_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- Interactive Controls ---
        self.dragged_node: Optional[HyperGraphNode] = None
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.after(16, self.animate)

    def _create_metric_row(self, label, var, color):
        frame = tk.Frame(self.sidebar, bg="#0f0f12")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, fg="#555", bg="#0f0f12", font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=var, fg=color, bg="#0f0f12", font=("Consolas", 9, "bold")).pack(side=tk.RIGHT)

    # --- Mouse Events ---
    
    def on_click(self, event):
        # Find closest node to click
        closest = None
        min_dist = float('inf')
        
        for node in self.kernel.nodes.values():
            dist = math.hypot(event.x - node.x, event.y - node.y)
            if dist < 20: # Hitbox radius
                if dist < min_dist:
                    min_dist = dist
                    closest = node
        
        if closest:
            self.dragged_node = closest
            self.dragged_node.is_dragging = True
            self.dragged_node.energy = 1.0 # Highlight on touch
            self.dragged_node.dx = 0
            self.dragged_node.dy = 0
    
    def on_drag(self, event):
        if self.dragged_node:
            self.dragged_node.x = event.x
            self.dragged_node.y = event.y

    def on_release(self, event):
        if self.dragged_node:
            self.dragged_node.is_dragging = False
            # Give a small random impulse on release
            self.dragged_node.dx = random.uniform(-0.5, 0.5)
            self.dragged_node.dy = random.uniform(-0.5, 0.5)
            self.dragged_node = None

    def animate(self):
        self.canvas.delete("all")
        
        nodes = list(self.kernel.nodes.values())
        
        # 1. Draw Connections
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                dist_sq = (n1.x - n2.x)**2 + (n1.y - n2.y)**2
                
                # Entanglement (Gold lines)
                if n1.is_entangled and n2.is_entangled and dist_sq < 90000:
                    self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#ffd700", width=1.5, dash=(4, 2))
                
                # Standard Connection (Blue-ish)
                elif dist_sq < 10000: 
                    alpha = int((1 - dist_sq/10000) * 100)
                    if alpha > 0:
                        color = f"#{alpha:02x}44{alpha:02x}"
                        self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00bcd4", width=1, stipple="gray50")

        # 2. Draw Nodes
        for node in nodes:
            # Color based on Quantum Phase (Rainbow cycle)
            rgb = colorsys.hsv_to_rgb(node.phase_angle / (2*math.pi), 0.7, 1.0)
            base_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            
            r = 3 + (node.energy * 6)
            
            if node.energy > 0.1:
                # Glow effect
                self.canvas.create_oval(node.x-r-3, node.y-r-3, node.x+r+3, node.y+r+3, outline=base_color, width=1)
                self.canvas.create_oval(node.x-r, node.y-r, node.x+r, node.y+r, fill="#ffffff", outline="")
            else:
                self.canvas.create_oval(node.x-r, node.y-r, node.x+r, node.y+r, fill="#222", outline=base_color)

            # Special visuals if being dragged
            if node.is_dragging:
                 # Draw "Cybernetic Target Lock"
                tr = r + 10
                self.canvas.create_arc(node.x-tr, node.y-tr, node.x+tr, node.y+tr, start=0, extent=60, style=tk.ARC, outline="#ff00ff", width=2)
                self.canvas.create_arc(node.x-tr, node.y-tr, node.x+tr, node.y+tr, start=180, extent=60, style=tk.ARC, outline="#ff00ff", width=2)
                self.canvas.create_line(node.x-tr-5, node.y, node.x-tr+5, node.y, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x+tr-5, node.y, node.x+tr+5, node.y, fill="#ff00ff", width=1)

        # 3. Update Sidebar Stats
        self.metrics_vars["status"].set(self.kernel.metrics["status"])
        self.metrics_vars["entropy"].set(f"{self.kernel.metrics['entropy']:.4f}")
        self.metrics_vars["coherence"].set(f"{self.kernel.metrics['coherence']:.1f}%")
        self.metrics_vars["born"].set(f"{self.kernel.metrics['born_prob']:.4f}")

        # 4. Update Logs
        if len(self.kernel.log_buffer) > 0:
            current_logs = list(self.kernel.log_buffer)
            self.log_list.delete(0, tk.END)
            for log in reversed(current_logs):
                self.log_list.insert(tk.END, log)
            
        self.after(16, self.animate)

# --- Entry Point ---

def run_gui():
    WIDTH, HEIGHT = 900, 650
    engine = AetheriusKernel(node_count=60, width=WIDTH, height=HEIGHT)
    engine.initialize_topology()
    
    def start_loop():
        asyncio.run(engine.run_simulation_loop())
        
    sim_thread = threading.Thread(target=start_loop, daemon=True)
    sim_thread.start()
    
    app = Visualizer(engine)
    app.mainloop()
    engine.running = False

if __name__ == "__main__":
    try:
        run_gui()
    except KeyboardInterrupt:
        pass