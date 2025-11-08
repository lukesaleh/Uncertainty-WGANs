import os, sys
import random
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
# Import Hugging Face datasets
from datasets import load_dataset
# Import the actual model from your models module
from src.models import Generator_CNN_Uncertainty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Config
# =========================
# Set cache directories to match training

# Get the project root directory (parent of src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "results", "uncertainty_nih", "samples")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "models_uncertainty", "cnn", "gan_uncertainty_xray_best_cnn.pth")
latent_dim = 128  # Match training config
img_size = 128    # Match training config

# Debug: Print paths
print(f"Project Root: {PROJECT_ROOT}")
print(f"Samples Dir: {SAMPLES_DIR}")
print(f"Model Path: {MODEL_PATH}")
print(f"Samples Dir Exists: {os.path.exists(SAMPLES_DIR)}")
print(f"Model Exists: {os.path.exists(MODEL_PATH)}")

# =========================
# Helper Functions
# =========================
# Global variable to cache the dataset
_cached_dataset = None

def load_nih_dataset():
    """Load NIH Chest X-ray dataset from Hugging Face cache"""
    global _cached_dataset
    
    if (_cached_dataset is not None):
        return _cached_dataset
    
    print("Loading NIH Chest X-ray dataset from cache...")
    
    try:
        # Load from cache (should be fast since data is already downloaded)
        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            name="image-classification",
            split="train",
            trust_remote_code=True
        )
        
        print(f"Dataset loaded: {len(dataset):,} images")
        print("Using subset of 5000 images for visualizer...")
        _cached_dataset = dataset.select(range(min(5000, len(dataset))))
        
        print(f"Dataset ready: {len(_cached_dataset)} images cached")
        return _cached_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the dataset cache exists at:")
        print(f"  {os.environ.get('HF_DATASETS_CACHE')}")
        return None

def get_available_epochs():
    """Get list of available epoch numbers from saved samples"""
    pattern = os.path.join(SAMPLES_DIR, "fake_epoch_*.png")
    files = glob.glob(pattern)
    epochs = []
    for f in files:
        basename = os.path.basename(f)
        # Extract number between "fake_epoch_" and ".png"
        if basename.startswith("fake_epoch_") and basename.endswith(".png"):
            try:
                epoch_str = basename[len("fake_epoch_"):-len(".png")]
                epoch_num = int(epoch_str)
                epochs.append(epoch_num)
            except ValueError:
                continue
    return sorted(epochs)

def load_generator():
    """Load the trained generator model"""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model not found at {MODEL_PATH}"
    
    try:
        G = Generator_CNN_Uncertainty(
            z_dim=latent_dim,
            img_channels=1,
            img_height=img_size,
            img_width=img_size
        ).to(device)
        
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint["G"]
        
        # Remove 'module.' prefix if present (from DataParallel/DDP)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict
        G.load_state_dict(state_dict, strict=False)
        G.eval()
        
        return G, f"Loaded model from epoch {checkpoint['epoch']}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error loading model: {str(e)}\n\nDetails:\n{error_details}"

def load_random_real_image():
    """Load a random real image from the Hugging Face NIH dataset"""
    try:
        dataset = load_nih_dataset()
        
        if dataset is None or len(dataset) == 0:
            print("Failed to load dataset")
            return None
        
        # Select random image
        idx = random.randint(0, len(dataset) - 1)
        example = dataset[idx]
        
        # Debug: print available keys
        print(f"Dataset example keys: {example.keys()}")
        
        # Try different possible image field names
        image = None
        for key in ['image', 'img', 'picture', 'x_ray']:
            if key in example:
                image = example[key]
                break
        
        if image is None:
            print(f"No image field found. Available fields: {list(example.keys())}")
            return None
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            print(f"Image type: {type(image)}, attempting conversion...")
            if hasattr(image, 'convert'):
                image = image.convert('L')
            else:
                return None
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply same transforms as training
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(image)
        print(f"Successfully loaded image with shape: {img_tensor.shape}")
        return img_tensor
        
    except Exception as e:
        import traceback
        print(f"Error in load_random_real_image: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_fake_image(generator):
    """Generate a fake image using the generator"""
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        fake_img = generator(z)
    return fake_img.squeeze()

# =========================
# GUI Application
# =========================
class GANVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uncertainty GAN Visualizer")
        self.root.geometry("900x700")
        
        # Variables
        self.generator = None
        self.current_answer = None
        self.score = 0
        self.total = 0
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_epoch_viewer_tab()
        self.create_game_tab()
        
    def create_epoch_viewer_tab(self):
        """Create the epoch sample viewer tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Epoch Viewer")
        
        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        ttk.Label(control_frame, text="Select Epoch:", font=('Arial', 12)).pack(side='left', padx=5)
        
        # Get available epochs
        self.epochs = get_available_epochs()
        
        if not self.epochs:
            ttk.Label(control_frame, text=f"No samples found in {SAMPLES_DIR}", 
                     foreground='red').pack(side='left', padx=5)
        else:
            self.epoch_var = tk.StringVar(value=str(self.epochs[0]))
            epoch_dropdown = ttk.Combobox(control_frame, textvariable=self.epoch_var, 
                                         values=[str(e) for e in self.epochs], 
                                         state='readonly', width=10)
            epoch_dropdown.pack(side='left', padx=5)
            
            ttk.Button(control_frame, text="View Samples", 
                      command=self.view_epoch).pack(side='left', padx=5)
            
            # Info label
            info_text = f"Available epochs: {len(self.epochs)} ({min(self.epochs)}-{max(self.epochs)})"
            ttk.Label(control_frame, text=info_text, font=('Arial', 9)).pack(side='left', padx=20)
        
        # Image display frame
        self.epoch_canvas_frame = ttk.Frame(tab)
        self.epoch_canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_game_tab(self):
        """Create the real or fake game tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Real or Fake Game")
        
        # Top frame for instructions and score
        top_frame = ttk.Frame(tab)
        top_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        ttk.Label(top_frame, text="Can you tell if the image is real or generated?", 
                 font=('Arial', 14, 'bold')).pack()
        
        self.score_label = ttk.Label(top_frame, text="Score: 0/0 (0.0%)", 
                                     font=('Arial', 12))
        self.score_label.pack(pady=5)
        
        # Load generator
        self.generator, msg = load_generator()
        status_color = 'green' if self.generator else 'red'
        ttk.Label(top_frame, text=msg, foreground=status_color, 
                 font=('Arial', 9)).pack()
        
        # Image display frame
        self.game_canvas_frame = ttk.Frame(tab)
        self.game_canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="New Round", command=self.new_game_round, 
                  state='normal' if self.generator else 'disabled').pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Real", command=lambda: self.make_guess('r'), 
                  state='disabled').pack(side='left', padx=5)
        self.real_button = control_frame.winfo_children()[-1]
        
        ttk.Button(control_frame, text="Fake", command=lambda: self.make_guess('f'), 
                  state='disabled').pack(side='left', padx=5)
        self.fake_button = control_frame.winfo_children()[-1]
        
        ttk.Button(control_frame, text="Reset Score", command=self.reset_score).pack(side='right', padx=5)
        
        self.result_label = ttk.Label(control_frame, text="", font=('Arial', 11, 'bold'))
        self.result_label.pack(side='left', padx=20)
        
    def view_epoch(self):
        """Display samples from selected epoch"""
        try:
            epoch = int(self.epoch_var.get())
            img_path = os.path.join(SAMPLES_DIR, f"fake_epoch_{epoch}.png")
            
            if not os.path.exists(img_path):
                messagebox.showerror("Error", f"Image not found: {img_path}")
                return
            
            # Clear previous canvas
            for widget in self.epoch_canvas_frame.winfo_children():
                widget.destroy()
            
            # Display image
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(f"Generated Samples - Epoch {epoch}", fontsize=14)
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=self.epoch_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def new_game_round(self):
        """Start a new round of the guessing game"""
        if not self.generator:
            messagebox.showerror("Error", "Generator not loaded")
            return
        
        # Clear previous canvas
        for widget in self.game_canvas_frame.winfo_children():
            widget.destroy()
        
        # Randomly decide real or fake
        self.current_answer = random.choice(['r', 'f'])
        
        try:
            if self.current_answer == 'r':
                img = load_random_real_image()
                if img is None:
                    messagebox.showerror("Error", "Could not load real image")
                    return
                img_display = img.cpu().numpy().squeeze()
            else:
                img = generate_fake_image(self.generator)
                img_display = img.cpu().numpy().squeeze()
            
            # Convert from [-1, 1] to [0, 1]
            img_display = (img_display + 1) / 2.0
            
            # Display image
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_display, cmap='gray')
            ax.set_title("Is this image REAL or FAKE?", fontsize=14, fontweight='bold')
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=self.game_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Enable guess buttons
            self.real_button.config(state='normal')
            self.fake_button.config(state='normal')
            self.result_label.config(text="Make your guess!", foreground='blue')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate image: {str(e)}")
    
    def make_guess(self, guess):
        """Process the user's guess"""
        if self.current_answer is None:
            return
        
        # Disable buttons
        self.real_button.config(state='disabled')
        self.fake_button.config(state='disabled')
        
        # Check answer
        correct = (guess == self.current_answer)
        self.total += 1
        
        if correct:
            self.score += 1
            self.result_label.config(text="✓ Correct!", foreground='green')
        else:
            answer_text = "REAL" if self.current_answer == 'r' else "FAKE"
            self.result_label.config(text=f"✗ Wrong! It was {answer_text}", foreground='red')
        
        # Update score
        percentage = (self.score / self.total * 100) if self.total > 0 else 0
        self.score_label.config(text=f"Score: {self.score}/{self.total} ({percentage:.1f}%)")
        
        self.current_answer = None
    
    def reset_score(self):
        """Reset the game score"""
        self.score = 0
        self.total = 0
        self.score_label.config(text="Score: 0/0 (0.0%)")
        self.result_label.config(text="")

# =========================
# Main
# =========================
def main():
    root = tk.Tk()
    app = GANVisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
