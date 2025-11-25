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
        
        # Friction if not gravitating (REDUCED FRICTION & ADDED THERMAL NOISE)
        if not self.is_gravitated:
            self.dx *= 0.998 # Slower decay
            self.dy *= 0.998
            
            # Thermal drift (keeps nodes moving)
            self.dx += random.uniform(-0.015, 0.015)
            self.dy += random.uniform(-0.015, 0.015)
        
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
        self.manual_instability = False # Toggle state
        self.entropy_history = [0.0] * 100 # For graph
        
        # Thread Safety Lock
        self.state_lock = threading.Lock()
        
        # New Feature: Singularity Engine
        self.singularity = {
            "enabled": True, # Toggle for the feature
            "active": False,
            "charge": 0.0,
            "phase": "IDLE",
            "timer": 0
        }
        
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
        
    def toggle_instability(self):
        """Toggles the manual instability mode."""
        with self.state_lock:
            self.manual_instability = not self.manual_instability
            
            if self.manual_instability:
                self.log("WARNING: MANUAL INSTABILITY TRIGGERED!")
                self.metrics["status"] = "SYSTEM UNSTABLE"
                # Initial Chaos Injection
                for node in self.nodes.values():
                    node.dx += random.uniform(-10.0, 10.0)
                    node.dy += random.uniform(-10.0, 10.0)
                    node.energy = 1.0
            else:
                self.log("INITIATING STABILIZATION PROTOCOLS...")
                self.metrics["status"] = "STABILIZING"
                # Dampen system
                for node in self.nodes.values():
                    node.dx *= 0.1
                    node.dy *= 0.1
                    node.energy = 0.0
                    node.quantum_state = QuantumState.SUPERPOSITION

    def toggle_singularity_mechanic(self):
        """Enables or disables the black hole generator."""
        with self.state_lock:
            self.singularity["enabled"] = not self.singularity["enabled"]
            state = "ENABLED" if self.singularity["enabled"] else "DISABLED"
            self.log(f"Singularity Generator {state}.")
            if not self.singularity["enabled"]:
                self.singularity["charge"] = 0.0
                self.singularity["active"] = False
                self.singularity["phase"] = "IDLE"

    def restore_system(self):
        """Hard reset of node physics with Thread Locking."""
        self.log("SYSTEM RESTORE INITIATED...")
        
        with self.state_lock:
            # Crucial Fix: Disable manual instability when restoring
            self.manual_instability = False
            self.metrics["status"] = "ONLINE"
            
            for node in self.nodes.values():
                node.x = random.randint(50, self.width - 50)
                node.y = random.randint(50, self.height - 50)
                node.dx = random.uniform(-0.5, 0.5)
                node.dy = random.uniform(-0.5, 0.5)
                node.energy = 0.0
                node.quantum_state = QuantumState.SUPERPOSITION
            
            self.singularity["charge"] = 0.0
            self.singularity["active"] = False
            self.singularity["phase"] = "IDLE"
            self.singularity["timer"] = 0
            
        self.log("System Normalized.")

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
            try:
                with self.state_lock:
                    # Physics Update
                    for node in self.nodes.values():
                        node.update_physics(self.width, self.height)
                        
                        # Active Instability Maintenance
                        if self.manual_instability:
                            if random.random() < 0.2:
                                node.dx += random.uniform(-2, 2)
                                node.dy += random.uniform(-2, 2)
                                node.energy = 1.0 # Force active to kill coherence
                    
                    # Handle Singularity Event Logic
                    if self.singularity["enabled"]:
                        if not self.singularity["active"]:
                            # Charge singularity based on entropy/instability
                            # Reduced growth rate for slower popup
                            base_growth = 0.0002 
                            instability_bonus = 0.001 if self.manual_instability else 0.0
                            self.singularity["charge"] = min(self.singularity["charge"] + base_growth + instability_bonus, 1.0)
                            
                            if self.singularity["charge"] >= 1.0:
                                self.singularity["active"] = True
                                self.singularity["phase"] = "ACCRETION"
                                self.singularity["timer"] = 240 # Frames (longer duration)
                                self.log("CRITICAL WARNING: SINGULARITY FORMING")
                        else:
                            self.singularity["timer"] -= 1
                            cx, cy = self.width / 2, self.height / 2
                            
                            if self.singularity["phase"] == "ACCRETION":
                                # Massive Gravity Well
                                for node in self.nodes.values():
                                    dx = cx - node.x
                                    dy = cy - node.y
                                    dist = math.hypot(dx, dy) + 1
                                    force = 250.0 / dist
                                    node.dx += (dx/dist) * force * 0.3
                                    node.dy += (dy/dist) * force * 0.3
                                
                                if self.singularity["timer"] <= 0:
                                    self.singularity["phase"] = "EXPULSION"
                                    self.singularity["timer"] = 40
                                    self.log("EVENT HORIZON BREACHED. EXPULSION IMMINENT.")

                            elif self.singularity["phase"] == "EXPULSION":
                                # Supernova Push
                                for node in self.nodes.values():
                                    dx = node.x - cx
                                    dy = node.y - cy
                                    dist = math.hypot(dx, dy) + 0.1
                                    force = 80.0 
                                    node.dx += (dx/dist) * force
                                    node.dy += (dy/dist) * force
                                    node.energy = 1.0
                                
                                if self.singularity["timer"] <= 0:
                                    self.singularity["active"] = False
                                    self.singularity["charge"] = 0.0
                                    self.singularity["phase"] = "IDLE"
                                    self.log("SINGULARITY COLLAPSED. FIELD NORMALIZED.")

                    # Quantum Tunneling
                    if random.random() < 0.05:
                        n1, n2 = random.sample(list(self.nodes.values()), 2)
                        await self.event_queue.put(QubitSignal("TUNNEL", n2.uid, 1.0, 0.0))

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
            except Exception as e:
                logger.error(f"Simulation Loop Error: {e}")
                await asyncio.sleep(0.1)

    def _update_metrics(self, cycle_id: int):
        active = sum(1 for n in self.nodes.values() if n.energy > 0.1)
        entropy = active / self.node_count
        
        self.metrics["entropy"] = entropy
        
        # Update History for Graph
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)

        # Calculate Coherence
        if self.manual_instability:
            self.metrics["coherence"] = random.uniform(-20.0, 15.0)
            self.metrics["status"] = "SYSTEM UNSTABLE"
        elif self.singularity["active"]:
            self.metrics["status"] = f"SINGULARITY: {self.singularity['phase']}"
        else:
            self.metrics["coherence"] = (1.0 - entropy) * 100
            if entropy > 0.8: self.metrics["status"] = "CRITICAL FLUX"
            elif entropy > 0.4: self.metrics["status"] = "OPTIMAL"
            else: self.metrics["status"] = "STABLE"

        self.metrics["born_prob"] = random.random()

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
        
        # Singularity Gauge
        self.lbl_singularity = tk.Label(self.sidebar, text="SINGULARITY CHARGE", fg="#aa00ff", bg="#0f0f12", font=("Consolas", 9))
        self.lbl_singularity.pack(anchor="w")
        self.charge_canvas = tk.Canvas(self.sidebar, height=10, bg="#1a0022", highlightthickness=0)
        self.charge_canvas.pack(fill=tk.X, pady=(2, 10))

        # Live Graph Canvas
        tk.Label(self.sidebar, text=":: LIVE TELEMETRY ::", fg="#555", bg="#0f0f12", font=("Consolas", 8)).pack(anchor="w", pady=(5, 0))
        self.graph_canvas = tk.Canvas(self.sidebar, height=60, bg="#0a0a10", highlightthickness=1, highlightbackground="#333")
        self.graph_canvas.pack(fill=tk.X, pady=(0, 20))

        # Controls Hint
        tk.Label(self.sidebar, text="[R-CLICK] Gravity Well", fg="#555", bg="#0f0f12", font=("Consolas", 8)).pack(anchor="w")
        tk.Label(self.sidebar, text="[SPACE] Toggle Instability", fg="#555", bg="#0f0f12", font=("Consolas", 8)).pack(anchor="w")
        tk.Label(self.sidebar, text="[B] Toggle Black Hole", fg="#555", bg="#0f0f12", font=("Consolas", 8)).pack(anchor="w")
        tk.Label(self.sidebar, text="[R] System Restore", fg="#555", bg="#0f0f12", font=("Consolas", 8)).pack(anchor="w", pady=(0,10))

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
        self.bind("<space>", self.on_space) 
        self.bind("b", self.on_b)
        self.bind("r", self.on_r)
        
        # Right Click (Gravity Well)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # --- Visual Effects State ---
        self.grid_offset = 0
        self.shockwaves = [] 
        self.particles = [] 
        self.data_packets = [] 
        self.lock_rotation = 0
        self.radar_angle = 0
        self.pulse_phase = 0.0
        
        # Transition Animation State
        self.transition_active = False
        self.transition_step = 0
        self.transition_color = ""

        # Generate Starfield
        self.stars = [
            {'x': random.randint(0, kernel.width), 
             'y': random.randint(0, kernel.height), 
             'z': random.uniform(0.5, 3.0),
             'size': random.uniform(0.5, 2.0)} 
            for _ in range(150)
        ]

        # Generate Nebula Clouds
        self.nebulae = [
            {'x': random.randint(0, kernel.width), 'y': random.randint(0, kernel.height), 
             'rx': random.randint(100, 300), 'ry': random.randint(50, 150),
             'color': random.choice(["#1a0b2e", "#0b1a2e", "#001010"])}
            for _ in range(5)
        ]
        
        self.after(16, self.animate)

    def _create_metric_row(self, label, var, color):
        frame = tk.Frame(self.sidebar, bg="#0f0f12")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, fg="#555", bg="#0f0f12", font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=var, fg=color, bg="#0f0f12", font=("Consolas", 9, "bold")).pack(side=tk.RIGHT)

    # --- Mouse & Key Events ---
    
    def on_click(self, event):
        closest = None
        min_dist = float('inf')
        
        # Use Lock for thread safety during interaction
        with self.kernel.state_lock:
            for node in self.kernel.nodes.values():
                dist = math.hypot(event.x - node.x, event.y - node.y)
                if dist < 25: 
                    if dist < min_dist:
                        min_dist = dist
                        closest = node
            
            if closest:
                self.dragged_node = closest
                self.dragged_node.is_dragging = True
                self.dragged_node.energy = 1.0 
                self.dragged_node.dx = 0
                self.dragged_node.dy = 0
    
    def on_drag(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Use Lock
        with self.kernel.state_lock:
            if self.dragged_node:
                self.dragged_node.x = event.x
                self.dragged_node.y = event.y
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
        with self.kernel.state_lock:
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
        
    def on_space(self, event):
        self.kernel.toggle_instability()
        self.transition_active = True
        self.transition_step = 25
        if self.kernel.manual_instability:
            self.transition_color = "#ff0000"
        else:
            self.transition_color = "#00ffff"

    def on_b(self, event):
        self.kernel.toggle_singularity_mechanic()
        if self.kernel.singularity["enabled"]:
            self.lbl_singularity.config(fg="#aa00ff", text="SINGULARITY CHARGE")
        else:
            self.lbl_singularity.config(fg="#555555", text="SINGULARITY [DISABLED]")

    def on_r(self, event):
        self.kernel.restore_system()
        self.transition_active = True
        self.transition_step = 25
        self.transition_color = "#ffffff"

    def draw_graph(self):
        self.graph_canvas.delete("all")
        width = self.graph_canvas.winfo_width()
        height = self.graph_canvas.winfo_height()
        data = self.kernel.entropy_history
        if not data: return
        
        # Scaling
        step_x = width / len(data)
        
        points = []
        for i, val in enumerate(data):
            x = i * step_x
            y = height - (val * height) 
            points.append(x)
            points.append(y)
        
        if len(points) >= 4:
            color = "#ff5252" if self.kernel.manual_instability else "#00ff9d"
            self.graph_canvas.create_line(points, fill=color, width=2)

    def draw_hex(self, x, y, radius, color, width=1):
        points = []
        for i in range(6):
            angle = math.radians(60 * i + self.lock_rotation * 2)
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.extend([px, py])
        self.canvas.create_polygon(points, outline=color, fill="", width=width)

    def animate(self):
        self.canvas.delete("all")
        w, h = self.kernel.width, self.kernel.height

        # -1. Draw Nebula Clouds (Background)
        for neb in self.nebulae:
            neb['x'] = (neb['x'] + 0.05) % (w + 200)
            self.canvas.create_oval(
                neb['x'] - neb['rx'], neb['y'] - neb['ry'],
                neb['x'] + neb['rx'], neb['y'] + neb['ry'],
                fill=neb['color'], outline=""
            )
        
        # 0. Draw Parallax Starfield
        mouse_dx = (self.mouse_x - w/2) * 0.02
        mouse_dy = (self.mouse_y - h/2) * 0.02
        
        for star in self.stars:
            sx = (star['x'] - mouse_dx * star['z']) % w
            sy = (star['y'] - mouse_dy * star['z']) % h
            alpha = int(100 * (1/star['z']))
            color = f"#{alpha:02x}{alpha:02x}{alpha:02x}"
            sz = star['size']
            self.canvas.create_oval(sx, sy, sx+sz, sy+sz, fill=color, outline="")

        # 1. Background Grid & Radar
        self.grid_offset = (self.grid_offset + 0.2) % 40
        self.radar_angle = (self.radar_angle + 2) % 360
        self.pulse_phase = (self.pulse_phase + 0.05) % (2 * math.pi)
        pulse_val = (math.sin(self.pulse_phase) + 1) * 0.5 # 0 to 1
        
        # Radar
        radar_x, radar_y = w / 2, h / 2
        self.canvas.create_arc(radar_x - 300, radar_y - 300, radar_x + 300, radar_y + 300, 
                               start=self.radar_angle, extent=30, style=tk.PIESLICE, 
                               outline="", fill="#0a0a14")

        # Grid
        for i in range(int(w / 40) + 2):
            x = i * 40 - (self.grid_offset)
            self.canvas.create_line(x, 0, x, h, fill="#111116", width=1)
        for i in range(int(h / 40) + 2):
            y = i * 40 + self.grid_offset - 40
            self.canvas.create_line(0, y, w, y, fill="#111116", width=1)

        # --- SINGULARITY VISUALS ---
        charge = self.kernel.singularity["charge"]
        
        # Update Charge Bar
        cw = self.charge_canvas.winfo_width()
        ch = self.charge_canvas.winfo_height()
        self.charge_canvas.delete("all")
        self.charge_canvas.create_rectangle(0, 0, cw * charge, ch, fill="#aa00ff", width=0)
        
        if self.kernel.singularity["enabled"]:
            # Visual precursor warning (80% - 100%)
            if charge > 0.8 and not self.kernel.singularity["active"]:
                # Destabilizing Field Effect - CYAN WARNING
                for _ in range(int(charge * 8)):
                    angle = random.uniform(0, 6.28)
                    dist = random.uniform(20, 150)
                    fx = radar_x + math.cos(angle) * dist
                    fy = radar_y + math.sin(angle) * dist
                    # Cyan lightning/cracks
                    self.canvas.create_line(fx, fy, radar_x, radar_y, fill="#00ffff", width=1, dash=(1,2))

            # Only draw event horizon if ACTIVE (charge reached 1.0)
            if self.kernel.singularity["active"]:
                radius = 60 + (random.randint(-2, 2))
                color = "#aa00ff" if self.kernel.singularity["phase"] == "ACCRETION" else "#ffffff"
                
                if self.kernel.singularity["phase"] == "EXPULSION":
                    radius *= 4 # Boom size
                    color = "#ffaa00"

                # Event Horizon
                self.canvas.create_oval(radar_x-radius, radar_y-radius, radar_x+radius, radar_y+radius, fill="#000000", outline=color, width=3)
                # Accretion Disk
                self.canvas.create_oval(radar_x-radius*2, radar_y-radius/2, radar_x+radius*2, radar_y+radius/2, outline=color, width=1, style=tk.ARC)
                self.canvas.create_oval(radar_x-radius/2, radar_y-radius*2, radar_x+radius/2, radar_y+radius*2, outline=color, width=1, style=tk.ARC)


        nodes = list(self.kernel.nodes.values())
        
        # Gravity Logic
        if self.gravity_active:
            self.canvas.create_oval(self.mouse_x-25, self.mouse_y-25, self.mouse_x+25, self.mouse_y+25, outline="#00bcd4", width=2, dash=(2,4))
            for node in nodes:
                dx = self.mouse_x - node.x
                dy = self.mouse_y - node.y
                dist = math.hypot(dx, dy)
                if dist > 10:
                    force = 60.0 / dist
                    node.dx += (dx / dist) * force * 0.5
                    node.dy += (dy / dist) * force * 0.5
                    node.is_gravitated = True
                else:
                    node.is_gravitated = False
        else:
            for node in nodes: node.is_gravitated = False

        # 2. Digital Glitch Effects
        if random.random() < 0.05:
            gx = random.randint(0, w)
            gy = random.randint(0, h)
            gw = random.randint(20, 150)
            gh = random.randint(2, 5)
            self.canvas.create_rectangle(gx, gy, gx+gw, gy+gh, fill="#00ff9d", stipple="gray25", outline="")

        # 3. Update & Draw Shockwaves
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

        # 4. Draw Connections & Data Packets
        new_packets = list(self.data_packets)
        
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                dist_sq = (n1.x - n2.x)**2 + (n1.y - n2.y)**2
                
                if n1.is_entangled and n2.is_entangled and dist_sq < 90000:
                    self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#ffd700", width=1.5, dash=(4, 2))
                elif dist_sq < 12000: 
                    alpha = int((1 - dist_sq/12000) * 100)
                    if alpha > 0:
                        self.canvas.create_line(n1.x, n1.y, n2.x, n2.y, fill="#00bcd4", width=1, stipple="gray50")
                        if random.random() < 0.005:
                            new_packets.append({
                                "x": n1.x, "y": n1.y, "tx": n2.x, "ty": n2.y,
                                "speed": 0.05 + random.random() * 0.05, "progress": 0.0
                            })

        active_packets = []
        for p in new_packets:
            p["progress"] += p["speed"]
            if p["progress"] < 1.0:
                cur_x = p["x"] + (p["tx"] - p["x"]) * p["progress"]
                cur_y = p["y"] + (p["ty"] - p["y"]) * p["progress"]
                self.canvas.create_rectangle(cur_x-1, cur_y-1, cur_x+1, cur_y+1, fill="white", outline="")
                active_packets.append(p)
        self.data_packets = active_packets

        # 5. Draw Nodes
        self.lock_rotation = (self.lock_rotation + 5) % 360
        
        for node in nodes:
            if node.just_collapsed:
                self.shockwaves.append({"x": node.x, "y": node.y, "radius": 5, "alpha": 1.0, "color": "#ff5252"})
                node.just_collapsed = False

            rgb = colorsys.hsv_to_rgb(node.phase_angle / (2*math.pi), 0.7, 1.0)
            base_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            
            # Pulse Effect
            pulse_size = pulse_val * 2
            r = 3 + (node.energy * 6) + pulse_size
            
            if node.energy > 0.1:
                # Ghosting Effect (Quantum Superposition)
                offset = random.randint(-2, 2)
                self.canvas.create_oval(node.x-r+offset, node.y-r+offset, node.x+r+offset, node.y+r+offset, outline=base_color, width=1, stipple="gray25")

                if random.random() < 0.02:
                     self.canvas.create_text(node.x + 15, node.y - 15, text=f"0x{random.randint(100,999)}", fill="#00ff9d", font=("Arial", 6))

                # ** NEW HEX-FORCEFIELD **
                self.draw_hex(node.x, node.y, r + 8, "#00ff9d")

                self.canvas.create_oval(node.x-r-3, node.y-r-3, node.x+r+3, node.y+r+3, outline=base_color, width=1)
                self.canvas.create_oval(node.x-r, node.y-r, node.x+r, node.y+r, fill="#ffffff", outline="")
            else:
                self.canvas.create_oval(node.x-r, node.y-r, node.x+r, node.y+r, fill="#222", outline=base_color)

            if node.is_dragging:
                tr = r + 12
                start_angle = self.lock_rotation
                self.canvas.create_arc(node.x-tr, node.y-tr, node.x+tr, node.y+tr, start=start_angle, extent=60, style=tk.ARC, outline="#ff00ff", width=2)
                self.canvas.create_arc(node.x-tr, node.y-tr, node.x+tr, node.y+tr, start=start_angle+180, extent=60, style=tk.ARC, outline="#ff00ff", width=2)
                self.canvas.create_line(node.x-tr-4, node.y, node.x-tr+4, node.y, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x+tr-4, node.y, node.x+tr+4, node.y, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x, node.y-tr-4, node.x, node.y-tr+4, fill="#ff00ff", width=1)
                self.canvas.create_line(node.x, node.y+tr-4, node.x, node.y+tr+4, fill="#ff00ff", width=1)

        # 6. Particles
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
        
        # 7. Scanlines
        for i in range(0, h, 4):
            self.canvas.create_line(0, i, w, i, fill="#080808", width=1)

        # === TRANSITION EFFECT (Screen Shake & Flash) ===
        if self.transition_active and self.transition_step > 0:
            # Screen Shake
            shake_intensity = self.transition_step
            shake_x = random.randint(-shake_intensity, shake_intensity)
            shake_y = random.randint(-shake_intensity, shake_intensity)
            self.canvas.move("all", shake_x, shake_y)

            # Flash Overlay
            stipple = "gray50" if self.transition_step % 4 < 2 else "gray25"
            self.canvas.create_rectangle(0, 0, w, h, fill=self.transition_color, stipple=stipple, outline="")
            
            self.transition_step -= 1
        else:
            self.transition_active = False

        # Update Sidebar
        self.metrics_vars["status"].set(self.kernel.metrics["status"])
        self.metrics_vars["entropy"].set(f"{self.kernel.metrics['entropy']:.4f}")
        
        # Format Coherence with glitch effect if negative
        coh = self.kernel.metrics['coherence']
        if coh < 0:
            self.metrics_vars["coherence"].set(f"ERR {coh:.1f}")
        else:
            self.metrics_vars["coherence"].set(f"{coh:.1f}%")
            
        self.metrics_vars["born"].set(f"{self.kernel.metrics['born_prob']:.4f}")

        # Update Logs
        if len(self.kernel.log_buffer) > 0:
            current_logs = list(self.kernel.log_buffer)
            self.log_list.delete(0, tk.END)
            for log in reversed(current_logs):
                self.log_list.insert(tk.END, log)
        
        # Draw Live Graph
        self.draw_graph()
            
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