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
        self.is_dragging = False
        self.is_gravitated = False
        
        # Quantum State (Logic)
        self._state_vector: StateVector = [complex(random.random(), random.random()) for _ in range(8)]
        self.quantum_state = QuantumState.SUPERPOSITION
        self.phase_angle = random.uniform(0, 2 * math.pi)
        self.is_entangled = False
        self.just_collapsed = False # Flag for shockwave generation

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
        
        # Friction if not gravitating
        if not self.is_gravitated:
            self.dx *= 0.99
            self.dy *= 0.99
        
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
        self.just_collapsed = True
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
        self.gravity_active = False
        self.mouse_x = 0
        self.mouse_y = 0
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Right Click (Gravity Well)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # --- Visual Effects State ---
        self.grid_offset = 0
        self.shockwaves = [] # List of dicts {x, y, radius, alpha}
        self.particles = [] # List of dicts {x, y, dx, dy, life}
        self.data_packets = [] # List of dicts {x, y, tx, ty, speed, color}
        self.lock_rotation = 0
        self.radar_angle = 0
        
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
            if dist < 25: # Hitbox radius
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
        self.mouse_x = event.x
        self.mouse_y = event.y
        if self.dragged_node:
            self.dragged_node.x = event.x
            self.dragged_node.y = event.y
            
            # Spawn particles while dragging
            if random.random() < 0.3:
                self.particles.append({
                    "x": event.x,
                    "y": event.y,
                    "dx": random.uniform(-2, 2),
                    "dy": random.uniform(-2, 2),
                    "life": 1.0,
                    "color": "#ff00ff"
                })

    def on_release(self, event):
        if self.dragged_node:
            self.dragged_node.is_dragging = False
            self.dragged_node.dx = random.uniform(-0.5, 0.5)
            self.dragged_node.dy = random.uniform(-0.5, 0.5)
            self.dragged_node = None
            
    def on_right_click(self, event):
        self.gravity_active = True
        self.mouse_x = event.x
        self.mouse_y = event.y
        
    def on_right_release(self, event):
        self.gravity_active = False
        
    def on_mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def animate(self):
        self.canvas.delete("all")
        
        # 0. Draw Background Grid & Radar
        self.grid_offset = (self.grid_offset + 0.2) % 40
        w, h = self.kernel.width, self.kernel.height
        
        # Radar Sweep
        self.radar_angle = (self.radar_angle + 2) % 360
        radar_x = w / 2
        radar_y = h / 2
        radar_len = max(w, h)
        rad = math.radians(self.radar_angle)
        # Using a polygon to simulate a scan sector
        self.canvas.create_arc(radar_x - 300, radar_y - 300, radar_x + 300, radar_y + 300, 
                               start=self.radar_angle, extent=30, style=tk.PIESLICE, 
                               outline="", fill="#0a0a14")

        # Grid Lines
        for i in range(int(w / 40) + 2):
            x = i * 40 - (self.grid_offset)
            self.canvas.create_line(x, 0, x, h, fill="#111116", width=1)
        for i in range(int(h / 40) + 2):
            y = i * 40 + self.grid_offset - 40
            self.canvas.create_line(0, y, w, y, fill="#111116", width=1)
            
        # Scanlines (Retro effect)
        for i in range(0, h, 4):
            self.canvas.create_line(0, i, w, i, fill="#080808", width=1)

        nodes = list(self.kernel.nodes.values())
        
        # Gravity Logic
        if self.gravity_active:
            self.canvas.create_oval(self.mouse_x-20, self.mouse_y-20, self.mouse_x+20, self.mouse_y+20, outline="#555", width=1, dash=(2,2))
            for node in nodes:
                dx = self.mouse_x - node.x
                dy = self.mouse_y - node.y
                dist = math.hypot(dx, dy)
                if dist > 10:
                    force = 50.0 / dist
                    node.dx += (dx / dist) * force * 0.5
                    node.dy += (dy / dist) * force * 0.5
                    node.is_gravitated = True
                else:
                    node.is_gravitated = False
        else:
            for node in nodes: node.is_gravitated = False

        # 1. Update & Draw Shockwaves
        new_shockwaves = []
        for wave in self.shockwaves:
            wave["radius"] += 3
            wave["alpha"] -= 0.04
            if wave["alpha"] > 0:
                color = wave["color"]
                stipple = "gray50" if wave["alpha"] < 0.5 else ""
                self.canvas.create_oval(
                    wave["x"] - wave["radius"], wave["y"] - wave["radius"],
                    wave["x"] + wave["radius"], wave["y"] + wave["radius"],
                    outline=color, width=2, stipple=stipple
                )
                new_shockwaves.append(wave)
        self.shockwaves = new_shockwaves

        # 2. Draw Connections & Data Packets
        new_packets = list(self.data_packets)
        
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
                        self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00bcd4", width=1, stipple="gray50")
                        
                        # Chance to spawn data packet
                        if random.random() < 0.005:
                            new_packets.append({
                                "x": n1.x, "y": n1.y,
                                "tx": n2.x, "ty": n2.y,
                                "speed": 0.05 + random.random() * 0.05,
                                "progress": 0.0
                            })

        # Update Packets
        active_packets = []
        for p in new_packets:
            p["progress"] += p["speed"]
            if p["progress"] < 1.0:
                cur_x = p["x"] + (p["tx"] - p["x"]) * p["progress"]
                cur_y = p["y"] + (p["ty"] - p["y"]) * p["progress"]
                self.canvas.create_rectangle(cur_x-1, cur_y-1, cur_x+1, cur_y+1, fill="white", outline="")
                active_packets.append(p)
        self.data_packets = active_packets

        # 3. Draw Nodes & Check Collapses
        self.lock_rotation = (self.lock_rotation + 5) % 360
        
        for node in nodes:
            # Check for collapse event (from kernel logic) to trigger visual shockwave
            if node.just_collapsed:
                self.shockwaves.append({"x": node.x, "y": node.y, "radius": 5, "alpha": 1.0, "color": "#ff5252"})
                node.just_collapsed = False # Reset flag

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
                 # Rotating Target Lock
                tr = r + 12
                start_angle = self.lock_rotation
                self.canvas.create_arc(node.x-tr, node.y-tr, node.x+tr, node.y+tr, start=start_angle, extent=60, style=tk.ARC, outline="#ff00ff", width=2)
                self.canvas.create_arc(node.x-tr, node.y-tr, node.x+tr, node.y+tr, start=start_angle+180, extent=60, style=tk.ARC, outline="#ff00ff", width=2)
                
                # Crosshairs
                self.canvas.create_line(node.x-tr-4, node.y, node.x-tr+4, node.y, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x+tr-4, node.y, node.x+tr+4, node.y, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x, node.y-tr-4, node.x, node.y-tr+4, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x, node.y+tr-4, node.x, node.y+tr+4, fill="#ff00ff", width=1)

        # 4. Update & Draw Particles
        new_particles = []
        for p in self.particles:
            p["x"] += p["dx"]
            p["y"] += p["dy"]
            p["life"] -= 0.05
            if p["life"] > 0:
                size = p["life"] * 3
                self.canvas.create_rectangle(p["x"]-size, p["y"]-size, p["x"]+size, p["y"]+size, fill=p["color"], outline="")
                new_particles.append(p)
        self.particles = new_particles

        # 5. Update Sidebar Stats
        self.metrics_vars["status"].set(self.kernel.metrics["status"])
        self.metrics_vars["entropy"].set(f"{self.kernel.metrics['entropy']:.4f}")
        self.metrics_vars["coherence"].set(f"{self.kernel.metrics['coherence']:.1f}%")
        self.metrics_vars["born"].set(f"{self.kernel.metrics['born_prob']:.4f}")

        # 6. Update Logs
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