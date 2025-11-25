import asyncio
import logging
import math
import random
import time
import uuid
import threading
import tkinter as tk
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
# Fixed the logging format to avoid Windows-specific errors
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
        self.x = random.randint(0, max_x)
        self.y = random.randint(0, max_y)
        self.dx = random.uniform(-1.5, 1.5)
        self.dy = random.uniform(-1.5, 1.5)
        self.energy = 0.0
        
        # Quantum State (Logic)
        self._state_vector: StateVector = [complex(random.random(), random.random()) for _ in range(8)]
        self.quantum_state = QuantumState.SUPERPOSITION

    def update_physics(self, width: int, height: int):
        self.x += self.dx
        self.y += self.dy
        
        # Wall bounce
        if self.x <= 0 or self.x >= width: self.dx *= -1
        if self.y <= 0 or self.y >= height: self.dy *= -1
        
        # Decay energy (for visual glow)
        self.energy *= 0.96

    @atomic_transaction(max_retries=5)
    async def integrate_signals(self, signals: List[complex]) -> None:
        if self.quantum_state == QuantumState.DECOHERENT: return
        self.energy = min(self.energy + 0.3, 1.0) # Boost visual energy
        await asyncio.sleep(0.001)

    @measure_execution_entropy
    def collapse_wavefunction(self) -> float:
        probabilities = [abs(c)**2 for c in self._state_vector]
        outcome = sum(p * i for i, p in enumerate(probabilities))
        self.quantum_state = QuantumState.COLLAPSED
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
        self.log_buffer = [] # Shared buffer for GUI logs
        self.connection_radius = 100

    def log(self, message: str):
        """Thread-safe logging to internal buffer and stdout"""
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

    async def _process_signal_propagation(self):
        while self.running:
            try:
                if not self.event_queue.empty():
                    signal: QubitSignal = await self.event_queue.get()
                    if signal.target_id in self.nodes:
                        await self.nodes[signal.target_id].integrate_signals([complex(signal.amplitude, signal.phase)])
                await asyncio.sleep(0.01)
            except Exception:
                pass

    async def run_simulation_loop(self):
        self.running = True
        self.log("Engaging Aetherius Kernel...")
        workers = [asyncio.create_task(self._process_signal_propagation()) for _ in range(3)]
        
        cycle = 0
        while self.running:
            # Physics Update
            for node in self.nodes.values():
                node.update_physics(self.width, self.height)
            
            # Random Stimulus
            if random.random() < 0.2:
                target = random.choice(list(self.nodes.values()))
                await self.event_queue.put(QubitSignal("EXT", target.uid, 1.0, 0.0))

            # Occasional Collapse
            if cycle % 60 == 0:
                self._observe_system(cycle)
            
            cycle += 1
            await asyncio.sleep(0.016) # ~60 FPS logic tick

    def _observe_system(self, cycle_id: int):
        active = sum(1 for n in self.nodes.values() if n.energy > 0.1)
        entropy = active / self.node_count
        if random.random() < 0.3:
            self.log(f"System Entropy: {entropy:.4f} J/K | Coherence: {(1-entropy)*100:.1f}%")

# --- Graphical User Interface (The "HTML-like" part) ---

class Visualizer(tk.Tk):
    def __init__(self, kernel: AetheriusKernel):
        super().__init__()
        self.kernel = kernel
        self.title("Aetherius Engine :: Quantum Visualizer")
        self.geometry(f"{kernel.width + 300}x{kernel.height}")
        self.configure(bg="#050505")
        
        # Layout
        self.canvas = tk.Canvas(self, width=kernel.width, height=kernel.height, bg="#000000", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        
        self.sidebar = tk.Frame(self, width=300, bg="#0f0f12")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Terminal Header
        tk.Label(self.sidebar, text=":: KERNEL LOGS ::", fg="#00ff9d", bg="#0f0f12", font=("Courier", 12, "bold")).pack(pady=10)
        
        # Log Box
        self.log_list = tk.Listbox(self.sidebar, bg="#0f0f12", fg="#a0a0a0", font=("Consolas", 9), borderwidth=0, highlightthickness=0)
        self.log_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Start Animation Loop
        self.after(16, self.animate)

    def animate(self):
        self.canvas.delete("all")
        
        # Draw Connections
        nodes = list(self.kernel.nodes.values())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                # Simple distance check
                dist_sq = (n1.x - n2.x)**2 + (n1.y - n2.y)**2
                if dist_sq < 10000: # 100^2 radius
                    alpha = int((1 - dist_sq/10000) * 100)
                    if alpha > 0:
                        color = f"#{alpha:02x}44{alpha:02x}" # Purple-ish tint
                        self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00bcd4", width=1, stipple="gray50")

        # Draw Nodes
        for node in nodes:
            r = 3 + (node.energy * 5)
            color = "#00ff9d" if node.energy > 0.1 else "#32323c"
            
            # Draw glow if active
            if node.energy > 0.1:
                self.canvas.create_oval(node.x-r-2, node.y-r-2, node.x+r+2, node.y+r+2, outline="#00ff9d", width=1)
            
            self.canvas.create_oval(node.x-r, node.y-r, node.x+r, node.y+r, fill=color, outline="")

        # Update Logs
        current_logs = list(self.kernel.log_buffer)
        self.log_list.delete(0, tk.END)
        for log in reversed(current_logs):
            self.log_list.insert(tk.END, log)
            
        self.after(16, self.animate)

# --- Entry Point ---

def run_gui():
    WIDTH, HEIGHT = 800, 600
    
    # 1. Initialize Kernel
    engine = AetheriusKernel(node_count=50, width=WIDTH, height=HEIGHT)
    engine.initialize_topology()
    
    # 2. Run Simulation in Background Thread
    def start_loop():
        asyncio.run(engine.run_simulation_loop())
        
    sim_thread = threading.Thread(target=start_loop, daemon=True)
    sim_thread.start()
    
    # 3. Run GUI in Main Thread
    app = Visualizer(engine)
    app.mainloop()
    
    # Cleanup
    engine.running = False

if __name__ == "__main__":
    try:
        run_gui()
    except KeyboardInterrupt:
        pass