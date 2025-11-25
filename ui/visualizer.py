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
# Add critic import
from src.models import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Config
# =========================
# Set cache directories to match training

# Get the project root directory (parent of src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define the best models directory
BEST_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "best")

# Function to find the correct model files
def find_model_path(model_type):
    """Find the model file in the best directory based on type"""
    if not os.path.exists(BEST_MODELS_DIR):
        print(f"Warning: Best models directory not found: {BEST_MODELS_DIR}")
        return None
    
    model_files = glob.glob(os.path.join(BEST_MODELS_DIR, "*.pth"))
    
    if model_type == 'Regular':
        # Find model WITHOUT 'finetuned' in the name
        for model_file in model_files:
            if 'finetuned' not in os.path.basename(model_file).lower():
                return model_file
    else:  # Finetuned
        # Find model WITH 'finetuned' in the name
        for model_file in model_files:
            if 'finetuned' in os.path.basename(model_file).lower():
                return model_file
    
    return None

# Define paths for both regular and finetuned models
MODEL_CONFIGS = {
    'Regular': {
        'samples_dir': os.path.join(PROJECT_ROOT, "results", "uncertainty_nih", "cnn", "samples"),
        'model_path': find_model_path('Regular'),
        'filename_pattern': "fake_epoch_*.png",
        'filename_extract': lambda f: int(f[len("fake_epoch_"):-len(".png")])
    },
    'Finetuned': {
        'samples_dir': os.path.join(PROJECT_ROOT, "results", "uncertainty_nih_finetuned", "cnn", "samples"),
        'model_path': find_model_path('Finetuned'),
        'filename_pattern': "finetuned_epoch_*_fid_*.png",
        'filename_extract': lambda f: int(f.split('_')[2])  # Extract epoch number from finetuned_epoch_015_fid_112.97.png
    }
}

SAMPLES_DIR = MODEL_CONFIGS['Regular']['samples_dir']
MODEL_PATH = MODEL_CONFIGS['Regular']['model_path']
latent_dim = 128  # Match training config
img_size = 128    # Match training config

# Debug: Print paths
print(f"Project Root: {PROJECT_ROOT}")
print(f"Best Models Dir: {BEST_MODELS_DIR}")
print(f"Regular Model Path: {MODEL_CONFIGS['Regular']['model_path']}")
print(f"Finetuned Model Path: {MODEL_CONFIGS['Finetuned']['model_path']}")
print(f"Samples Dir: {SAMPLES_DIR}")
print(f"Best Models Dir Exists: {os.path.exists(BEST_MODELS_DIR)}")
print(f"Regular Model Exists: {MODEL_CONFIGS['Regular']['model_path'] and os.path.exists(MODEL_CONFIGS['Regular']['model_path'])}")
print(f"Finetuned Model Exists: {MODEL_CONFIGS['Finetuned']['model_path'] and os.path.exists(MODEL_CONFIGS['Finetuned']['model_path'])}")

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
            trust_remote_code=True,
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

def get_available_epochs(model_type='Regular'):
    """Get list of available epoch numbers from saved samples"""
    config = MODEL_CONFIGS[model_type]
    pattern = os.path.join(config['samples_dir'], config['filename_pattern'])
    files = glob.glob(pattern)
    epochs = []
    for f in files:
        basename = os.path.basename(f)
        try:
            epoch_num = config['filename_extract'](basename)
            epochs.append(epoch_num)
        except (ValueError, IndexError):
            continue
    return sorted(epochs)

def load_models(model_type='Regular'):
    """Load both generator and critic models"""
    model_path = MODEL_CONFIGS[model_type]['model_path']
    
    if not os.path.exists(model_path):
        return None, None, f"Model not found at {model_path}"
    
    try:
        G = Generator_CNN_Uncertainty(
            z_dim=latent_dim,
            img_channels=1,
            img_height=img_size,
            img_width=img_size
        ).to(device)
        
        D = Critic(
            img_channels=1,
            img_height=img_size,
            img_width=img_size
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Load generator
        g_state_dict = checkpoint["G"]
        if list(g_state_dict.keys())[0].startswith('module.'):
            g_state_dict = {k.replace('module.', ''): v for k, v in g_state_dict.items()}
        G.load_state_dict(g_state_dict, strict=False)
        
        # Load critic
        d_state_dict = checkpoint["D"]
        if list(d_state_dict.keys())[0].startswith('module.'):
            d_state_dict = {k.replace('module.', ''): v for k, v in d_state_dict.items()}
        D.load_state_dict(d_state_dict, strict=False)
        
        G.eval()
        D.eval()
        
        return G, D, f"Loaded {model_type} model from epoch {checkpoint['epoch']}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, None, f"Error loading model: {str(e)}\n\nDetails:\n{error_details}"

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

def get_epoch_filename(epoch, model_type='Regular'):
    """Get the filename for a given epoch and model type"""
    config = MODEL_CONFIGS[model_type]
    samples_dir = config['samples_dir']
    
    if model_type == 'Regular':
        return os.path.join(samples_dir, f"fake_epoch_{epoch}.png")
    else:  # Finetuned
        # Find the file that matches the epoch
        pattern = os.path.join(samples_dir, f"finetuned_epoch_{epoch:03d}_fid_*.png")
        files = glob.glob(pattern)
        if files:
            return files[0]
        return None

# =========================
# GUI Application
# =========================
class GANVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uncertainty GAN Visualizer")
        self.root.geometry("900x700")
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables
        self.generator = None
        self.critic = None  # Add critic
        self.current_answer = None
        self.current_image = None  # Store current image for critic scoring
        self.score = 0
        self.total = 0
        self.current_model_type = 'Regular'
        
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
        
        # Model type selector
        ttk.Label(control_frame, text="Model Type:", font=('Arial', 12)).pack(side='left', padx=5)
        self.model_type_var = tk.StringVar(value='Regular')
        model_type_dropdown = ttk.Combobox(control_frame, textvariable=self.model_type_var,
                                          values=['Regular', 'Finetuned'],
                                          state='readonly', width=12)
        model_type_dropdown.pack(side='left', padx=5)
        model_type_dropdown.bind('<<ComboboxSelected>>', self.on_model_type_changed)
        
        ttk.Label(control_frame, text="Select Epoch:", font=('Arial', 12)).pack(side='left', padx=(20, 5))
        
        # Get available epochs
        self.epochs = get_available_epochs('Regular')
        
        if not self.epochs:
            ttk.Label(control_frame, text=f"No samples found", 
                     foreground='red').pack(side='left', padx=5)
        else:
            self.epoch_var = tk.StringVar(value=str(self.epochs[0]))
            self.epoch_dropdown = ttk.Combobox(control_frame, textvariable=self.epoch_var, 
                                         values=[str(e) for e in self.epochs], 
                                         state='readonly', width=10)
            self.epoch_dropdown.pack(side='left', padx=5)
            
            ttk.Button(control_frame, text="View Samples", 
                      command=self.view_epoch).pack(side='left', padx=5)
            
            # Info label
            info_text = f"Available epochs: {len(self.epochs)} ({min(self.epochs)}-{max(self.epochs)})"
            self.epoch_info_label = ttk.Label(control_frame, text=info_text, font=('Arial', 9))
            self.epoch_info_label.pack(side='left', padx=20)
        
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
        
        # Add critic score label
        self.critic_score_label = ttk.Label(top_frame, text="Critic Score: --", 
                                           font=('Arial', 11))
        self.critic_score_label.pack(pady=2)
        
        # Model type selector for game
        model_selector_frame = ttk.Frame(top_frame)
        model_selector_frame.pack(pady=5)
        
        ttk.Label(model_selector_frame, text="Generator:", font=('Arial', 10)).pack(side='left', padx=5)
        self.game_model_type_var = tk.StringVar(value='Regular')
        game_model_dropdown = ttk.Combobox(model_selector_frame, textvariable=self.game_model_type_var,
                                          values=['Regular', 'Finetuned'],
                                          state='readonly', width=12)
        game_model_dropdown.pack(side='left', padx=5)
        game_model_dropdown.bind('<<ComboboxSelected>>', self.on_game_model_type_changed)
        
        # Load both generator and critic
        self.generator, self.critic, msg = load_models('Regular')
        status_color = 'green' if self.generator else 'red'
        self.game_status_label = ttk.Label(top_frame, text=msg, foreground=status_color, 
                 font=('Arial', 9))
        self.game_status_label.pack()
        
        # Image display frame
        self.game_canvas_frame = ttk.Frame(tab)
        self.game_canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        
        self.new_round_button = ttk.Button(control_frame, text="New Round", command=self.new_game_round, 
                  state='normal' if self.generator else 'disabled')
        self.new_round_button.pack(side='left', padx=5)
        
        self.real_button = ttk.Button(control_frame, text="Real", command=lambda: self.make_guess('r'), 
                  state='disabled')
        self.real_button.pack(side='left', padx=5)
        
        self.fake_button = ttk.Button(control_frame, text="Fake", command=lambda: self.make_guess('f'), 
                  state='disabled')
        self.fake_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Reset Score", command=self.reset_score).pack(side='right', padx=5)
        
        self.result_label = ttk.Label(control_frame, text="", font=('Arial', 11, 'bold'))
        self.result_label.pack(side='left', padx=20)
    
    def on_model_type_changed(self, event=None):
        """Handle model type selection change in epoch viewer"""
        model_type = self.model_type_var.get()
        self.epochs = get_available_epochs(model_type)
        
        if self.epochs:
            self.epoch_var.set(str(self.epochs[0]))
            self.epoch_dropdown['values'] = [str(e) for e in self.epochs]
            info_text = f"Available epochs: {len(self.epochs)} ({min(self.epochs)}-{max(self.epochs)})"
            self.epoch_info_label.config(text=info_text)
        else:
            self.epoch_var.set("")
            self.epoch_dropdown['values'] = []
            self.epoch_info_label.config(text="No samples found")
    
    def on_game_model_type_changed(self, event=None):
        """Handle model type selection change in game tab"""
        model_type = self.game_model_type_var.get()
        self.generator, self.critic, msg = load_models(model_type)
        status_color = 'green' if self.generator else 'red'
        self.game_status_label.config(text=msg, foreground=status_color)
        
        # Enable/disable new round button
        self.new_round_button.config(state='normal' if self.generator else 'disabled')
        
    def get_critic_score(self, img_tensor):
        """Compute critic score for an image"""
        if self.critic is None:
            return None
        
        with torch.no_grad():
            # Add batch dimension if needed
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            score = self.critic(img_tensor).item()
        return score
    
    def view_epoch(self):
        """Display samples from selected epoch"""
        try:
            epoch = int(self.epoch_var.get())
            model_type = self.model_type_var.get()
            img_path = get_epoch_filename(epoch, model_type)
            
            if img_path is None or not os.path.exists(img_path):
                messagebox.showerror("Error", f"Image not found for epoch {epoch}")
                return
            
            # Clear previous canvas
            for widget in self.epoch_canvas_frame.winfo_children():
                widget.destroy()
            
            # Display image
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(f"{model_type} Generated Samples - Epoch {epoch}", fontsize=14)
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
                self.current_image = img.unsqueeze(0).to(device)  # Store for critic
                img_display = img.cpu().numpy().squeeze()
            else:
                img = generate_fake_image(self.generator)
                self.current_image = img.unsqueeze(0).to(device)  # Store for critic
                img_display = img.cpu().numpy().squeeze()
            
            # Get critic score
            critic_score = self.get_critic_score(self.current_image)
            if critic_score is not None:
                # Higher scores = more "real" according to critic
                self.critic_score_label.config(
                    text=f"Critic Score: {critic_score:.3f} ({'Real-like' if critic_score > 0 else 'Fake-like'})",
                    foreground='blue'
                )
            else:
                self.critic_score_label.config(text="Critic Score: --", foreground='gray')
            
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
        
        # Get critic score for feedback
        critic_score = self.get_critic_score(self.current_image) if self.current_image is not None else None
        
        if correct:
            self.score += 1
            result_text = "✓ Correct!"
            if critic_score is not None:
                critic_verdict = "agreed" if (guess == 'r' and critic_score > 0) or (guess == 'f' and critic_score < 0) else "disagreed"
                result_text += f" (Critic {critic_verdict})"
            self.result_label.config(text=result_text, foreground='green')
        else:
            answer_text = "REAL" if self.current_answer == 'r' else "FAKE"
            result_text = f"✗ Wrong! It was {answer_text}"
            if critic_score is not None:
                critic_correct = (self.current_answer == 'r' and critic_score > 0) or (self.current_answer == 'f' and critic_score < 0)
                result_text += f" (Critic was {'right' if critic_correct else 'wrong'})"
            self.result_label.config(text=result_text, foreground='red')
        
        # Update critic score label with final verdict
        if critic_score is not None:
            self.critic_score_label.config(
                text=f"Critic Score: {critic_score:.3f} → Predicted: {'REAL' if critic_score > 0 else 'FAKE'}",
                foreground='green' if (critic_score > 0 and self.current_answer == 'r') or (critic_score < 0 and self.current_answer == 'f') else 'red'
            )
        
        # Update score label
        percentage = (self.score / self.total * 100) if self.total > 0 else 0
        self.score_label.config(text=f"Score: {self.score}/{self.total} ({percentage:.1f}%)")

        self.current_answer = None
    
    def reset_score(self):
        """Reset the game score"""
        self.score = 0
        self.total = 0
        self.score_label.config(text="Score: 0/0 (0.0%)")
        self.result_label.config(text="")

    def on_closing(self):
        """Handle window closing"""
        self.root.destroy()
        sys.exit()

# =========================
# Main
# =========================
def main():
    root = tk.Tk()
    app = GANVisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
