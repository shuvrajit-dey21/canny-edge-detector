# Subject :- Computer vision 
# Assignment Topic :- To implement Canny Edge detection Algorithm 
# Name :- Shuvrajit Dey
# ID No:- 22IUT0010170


# Import necessary libraries for GUI and image processing
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import math
import time

# Class to display algorithm information in a separate window
class AlgorithmInfoWindow:
    def __init__(self, parent):
        # Create a new window with fade-in effect
        self.window = tk.Toplevel(parent)
        self.window.title("Canny Edge Detection - Algorithm Information")
        self.window.geometry("800x600")
        self.window.configure(bg='#f0f0f0')
        
        # Make the window float on top of the parent window
        self.window.transient(parent)
        self.window.grab_set()
        
        # Set initial transparency for fade-in effect
        self.window.attributes('-alpha', 0.0)
        
        # Configure styles for labels and frames
        style = ttk.Style()
        style.configure('Info.TLabel', 
                       font=('Helvetica', 11),
                       background='#f0f0f0',
                       wraplength=700)
        style.configure('InfoTitle.TLabel',
                       font=('Helvetica', 14, 'bold'),
                       background='#f0f0f0',
                       foreground='#2c3e50')
        style.configure('InfoSection.TFrame',
                       background='#ffffff',
                       relief='solid')
        style.configure('Hover.TFrame',
                       background='#e8f0fe')
        
        # Create main frame with custom canvas for smooth scrolling
        main_frame = ttk.Frame(self.window, style='InfoSection.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create canvas with custom scrolling
        self.canvas = tk.Canvas(main_frame,
                              bg='#ffffff',
                              highlightthickness=0,
                              relief='flat')
        scrollbar = ttk.Scrollbar(main_frame,
                                orient="vertical",
                                command=self.smooth_scroll)
        self.content_frame = ttk.Frame(self.canvas,
                                     style='InfoSection.TFrame')
        
        # Configure scrolling
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas
        self.canvas_frame = self.canvas.create_window(
            (0, 0),
            window=self.content_frame,
            anchor="nw",
            width=self.canvas.winfo_reqwidth()
        )
        
        # Add sections with animation delays
        self.sections = [
            ("Canny Edge Detection Algorithm",
             """The Canny edge detection algorithm is a multi-stage algorithm developed by John F. Canny in 1986. 
             It is considered one of the most robust edge detection algorithms."""),
            ("1. Grayscale Conversion",
             """Convert the image to grayscale using weighted sum:
             gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
             
             This weights are based on human perception of color."""),
            ("2. Gaussian Blur",
             """Apply Gaussian blur to reduce noise:
             - Create 5x5 Gaussian kernel using the formula:
               G(x,y) = (1/2πσ²)e^(-(x²+y²)/2σ²)
             - Convolve image with kernel
             - Reduces noise while preserving edges"""),
            ("3. Gradient Calculation",
             """Calculate intensity gradients:
             - Apply Sobel operators in x and y directions
             - Find gradient magnitude: √(Gx² + Gy²)
             - Find gradient direction: θ = arctan(Gy/Gx)
             
             Sobel operators:
             X = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
             
             Y = [[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]]"""),
            ("4. Non-Maximum Suppression",
             """Thin edges by suppressing non-maximum values:
             1. Round gradient direction to nearest 45°
             2. Compare with pixels in gradient direction
             3. Suppress if not local maximum
             
             This creates thin, precise edges."""),
            ("5. Double Thresholding",
             """Identify strong and weak edges:
             - High threshold (strong): typically 0.15 * max
             - Low threshold (weak): typically 0.05 * max
             
             Creates three categories:
             - Strong edges (keep)
             - Weak edges (evaluate)
             - Non-edges (discard)"""),
            ("6. Edge Tracking by Hysteresis",
             """Connect edges using hysteresis:
             1. Start with strong edges
             2. Recursively add connected weak edges
             3. Remove isolated weak edges
             
             This creates continuous edge lines.""")
        ]
        
        # Add sections with animation
        self.section_frames = []
        for i, (title, content) in enumerate(self.sections):
            self.window.after(i * 100, lambda t=title, c=content: 
                            self.add_section_with_animation(t, c))
        
        # Configure canvas scrolling
        self.content_frame.bind('<Configure>', self.on_frame_configure)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Bind mouse wheel for smooth scrolling
        self.canvas.bind_all('<MouseWheel>', self.on_mousewheel)
        
        # Close button with hover effect
        self.close_btn = ttk.Button(self.window,
                                  text="Close",
                                  command=self.close_with_animation,
                                  style='Custom.TButton')
        self.close_btn.pack(pady=10)
        
        # Start fade-in animation
        self.fade_in()

    def fade_in(self, alpha=0.0):
        """Animate window fade in"""
        if alpha < 1.0:
            alpha += 0.1
            self.window.attributes('-alpha', alpha)
            self.window.after(20, lambda: self.fade_in(alpha))

    def fade_out(self, alpha=1.0):
        """Animate window fade out"""
        if alpha > 0:
            alpha -= 0.1
            self.window.attributes('-alpha', alpha)
            self.window.after(20, lambda: self.fade_out(alpha))
        else:
            self.window.destroy()

    def close_with_animation(self):
        """Close window with fade-out animation"""
        self.fade_out()

    def add_section_with_animation(self, title, content):
        """Add a section with slide-in animation"""
        frame = ttk.Frame(self.content_frame, style='InfoSection.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=10)
        frame.pack_propagate(False)  # Prevent size changes
        
        # Create content
        title_label = ttk.Label(frame, text=title, style='InfoTitle.TLabel')
        title_label.pack(anchor='w', pady=(5, 0))
        
        content_label = ttk.Label(frame, text=content, style='Info.TLabel')
        content_label.pack(anchor='w', pady=(5, 10))
        
        # Add hover effect
        frame.bind('<Enter>', lambda e: self.on_section_hover(frame, True))
        frame.bind('<Leave>', lambda e: self.on_section_hover(frame, False))
        
        # Animate frame height
        frame.update()
        required_height = title_label.winfo_reqheight() + content_label.winfo_reqheight() + 20
        frame.configure(height=1)
        self.animate_frame_height(frame, required_height)
        
        self.section_frames.append(frame)

    def animate_frame_height(self, frame, target_height, current_height=1):
        """Animate frame height smoothly"""
        if current_height < target_height:
            current_height += (target_height - current_height) * 0.2
            if current_height < target_height - 1:
                frame.configure(height=int(current_height))
                self.window.after(10, lambda: self.animate_frame_height(frame, target_height, current_height))
            else:
                frame.configure(height=target_height)
                frame.pack_propagate(True)

    def on_section_hover(self, frame, entering):
        """Handle section hover effect"""
        frame.configure(style='Hover.TFrame' if entering else 'InfoSection.TFrame')

    def smooth_scroll(self, *args):
        """Implement smooth scrolling"""
        if len(args) > 1:
            self.canvas.yview_moveto(args[1])
        else:
            # Use smoother scrolling with acceleration
            amount = int(args[0])
            # Apply scrolling with acceleration effect
            if amount != 0:
                for i in range(3):
                    factor = 0.7 ** i  # Decreasing factor for deceleration
                    scroll_amount = int(amount * factor) if amount * factor >= 1 or amount * factor <= -1 else amount
                    self.window.after(i * 5, lambda a=scroll_amount: self.canvas.yview_scroll(a, 'units'))

    def on_mousewheel(self, event):
        """Handle smooth mousewheel scrolling with improved animation"""
        # Get the delta value and normalize it
        delta = -1 * (event.delta // 120)
        
        # Use more steps with smaller increments for smoother animation
        steps = 15  # Increased steps for smoother animation
        
        # Apply scrolling with cubic deceleration curve
        for i in range(steps):
            factor = 1 - (i / steps) ** 3  # Cubic deceleration for smoother stop
            scroll_amount = int(delta * 2 * factor)  # Multiply by 2 for better initial momentum
            scroll_amount = max(1, scroll_amount) if scroll_amount > 0 else min(-1, scroll_amount)
            # Apply with increasing delay for natural deceleration
            self.window.after(i * 5, lambda a=scroll_amount: self.canvas.yview_scroll(a, 'units'))

    def on_frame_configure(self, event=None):
        """Reset scroll region when content frame size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Update canvas window size when canvas is resized"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

# Main class for the Canny Edge Detection tool
class CannyEdgeDetector:
    def __init__(self):
        # Initialize the main window
        self.window = tk.Tk()
        self.window.title("Canny Edge Detection Tool")
        self.window.geometry("1200x800")
        self.window.configure(bg='#f0f0f0')
        
        # Set theme for the application
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles for frames and buttons
        style.configure('Custom.TFrame', background='#f0f0f0')
        style.configure('Custom.TButton',
                       padding=10,
                       font=('Helvetica', 10, 'bold'))
        style.configure('Title.TLabel',
                       font=('Helvetica', 28, 'bold'),  # Increased font size
                       background='#f0f0f0',
                       foreground='#2c3e50')
        style.configure('Subtitle.TLabel',
                       font=('Helvetica', 12),
                       background='#f0f0f0',
                       foreground='#34495e')
        style.configure('Progress.Horizontal.TProgressbar',
                       background='#2ecc71',
                       troughcolor='#ecf0f1',
                       bordercolor='#bdc3c7')
        
        # Create main canvas for scrolling
        self.main_canvas = tk.Canvas(self.window, bg='#f0f0f0', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=self.smooth_scroll)
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # Create main frame inside canvas
        self.main_frame = ttk.Frame(self.main_canvas, padding="20", style='Custom.TFrame')
        self.canvas_frame = self.main_canvas.create_window(
            (0, 0),
            window=self.main_frame,
            anchor="nw",
            width=self.main_canvas.winfo_reqwidth()
        )
        
        # Title and buttons frame
        title_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        title_frame.grid(row=0, column=0, columnspan=2, sticky='ew')
        title_frame.grid_columnconfigure(0, weight=1)  # Make middle column expandable
        
        # Title (centered)
        self.title_label = ttk.Label(title_frame,
                                   text="Canny Edge Detection",
                                   style='Title.TLabel',
                                   anchor='center')
        self.title_label.grid(row=0, column=0, pady=(0, 20), sticky='ew')  # Centered title
        
        # Button frame
        self.button_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons (all in one line)
        self.choose_btn = ttk.Button(self.button_frame,
                                   text="Choose Image",
                                   command=self.load_image,
                                   style='Custom.TButton')
        self.choose_btn.grid(row=0, column=0, padx=10)
        
        self.process_btn = ttk.Button(self.button_frame,
                                    text="Process",
                                    command=self.process_image_with_progress,
                                    style='Custom.TButton')
        self.process_btn.grid(row=0, column=1, padx=10)
        
        self.reset_btn = ttk.Button(self.button_frame,
                                  text="Reset",
                                  command=self.reset_images,
                                  style='Custom.TButton')
        self.reset_btn.grid(row=0, column=2, padx=10)
        
        # Info button (next to reset button)
        self.info_btn = ttk.Button(self.button_frame,
                                 text="ℹ️ Algorithm Info",
                                 command=self.show_algorithm_info,
                                 style='Custom.TButton')
        self.info_btn.grid(row=0, column=3, padx=10)
        
        # Progress frame
        self.progress_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.progress_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                          variable=self.progress_var,
                                          maximum=100,
                                          mode='determinate',
                                          length=400,
                                          style='Progress.Horizontal.TProgressbar')
        self.progress_bar.grid(row=0, column=0, padx=(0, 10))
        
        # Progress percentage label
        self.progress_label = ttk.Label(self.progress_frame,
                                      text="0%",
                                      style='Subtitle.TLabel')
        self.progress_label.grid(row=0, column=1)
        
        # Hide progress frame initially
        self.progress_frame.grid_remove()
        
        # Status label
        self.status_label = ttk.Label(self.main_frame,
                                    text="",
                                    style='Subtitle.TLabel')
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Image frames
        self.create_image_frame("Original Image", 4, 0)
        self.create_image_frame("Edge Detected Image", 4, 1)
        
        # Initialize variables
        self.current_image = None
        self.processed_image = None
        
        # Load default square image
        self.create_default_square_image()
        
        # Configure grid weights for responsive layout
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(4, weight=1)  # Make image frames expandable
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Bind events for smooth scrolling with improved responsiveness
        self.main_canvas.bind('<Configure>', self.on_canvas_configure)
        self.main_frame.bind('<Configure>', self.on_frame_configure)
        self.main_canvas.bind_all('<MouseWheel>', self.on_mousewheel)
        
        # Ensure buttons stay visible during processing
        self.button_frame.lift()
        self.progress_frame.lift()

    def create_default_square_image(self):
        """Create default square image as shown in screenshot"""
        # Create larger image to better fill the frame
        size = 400  # Increased size
        img = Image.new('RGB', (size, size), 'black')
        draw = ImageDraw.Draw(img)
        
        # Calculate sizes for squares
        outer_size = int(size * 0.6)  # 60% of total size
        inner_size = int(outer_size * 0.3)  # 30% of outer square
        
        # Calculate positions
        outer_offset = (size - outer_size) // 2
        outer_box = [(outer_offset, outer_offset),
                    (outer_offset + outer_size, outer_offset + outer_size)]
        
        # Draw outer white square
        draw.rectangle(outer_box, fill='white')
        
        # Calculate inner square position
        inner_offset = (size - inner_size) // 2
        inner_box = [(inner_offset, inner_offset),
                    (inner_offset + inner_size, inner_offset + inner_size)]
        
        # Draw inner black square
        draw.rectangle(inner_box, fill='black')
        
        self.current_image = np.array(img)
        self.display_image(img, self.original_image_label)

    def show_algorithm_info(self):
        """Show algorithm information window"""
        AlgorithmInfoWindow(self.window)

    def reset_images(self):
        """Reset to default square image"""
        self.create_default_square_image()
        if self.processed_image_label:
            self.processed_image_label.configure(image='')
        self.hide_progress()
        self.status_label.config(text="Reset complete")

    def update_progress(self, value, status_text):
        """Update progress bar, percentage and status text"""
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")
        self.status_label.config(text=status_text)
        self.window.update()

    def hide_progress(self):
        """Hide progress elements"""
        self.progress_frame.grid_remove()
        self.status_label.config(text="")
        self.progress_var.set(0)
        self.progress_label.config(text="0%")

    def create_image_frame(self, title, row, column):
        """Create a frame for displaying images with a title and border"""
        # Create main frame with fixed size and gray background
        frame = ttk.Frame(self.main_frame, padding="10", relief="solid", borderwidth=1)
        frame.grid(row=row, column=column, padx=5, pady=(0, 10), sticky="nsew")
        
        # Force both frames to have identical width
        self.main_frame.grid_columnconfigure(0, weight=1, uniform="equal")  # Use uniform to ensure equal size
        self.main_frame.grid_columnconfigure(1, weight=1, uniform="equal")  # Use uniform to ensure equal size
        
        # Configure frame size with optimal dimensions for small screens
        frame.grid_propagate(False)  # Prevent frame from resizing to content
        frame.configure(width=350, height=350)  # Reduced dimensions for better small screen compatibility
        
        # Title with enhanced styling
        style = ttk.Style()
        style.configure('FrameTitle.TLabel',
                       font=('Helvetica', 14, 'bold'),
                       foreground='#2c3e50',
                       background='#e0e0e0',
                       padding=5)
        
        # Title container frame with background
        title_container = ttk.Frame(frame, style='Custom.TFrame')
        title_container.pack(fill="x", pady=(0, 10))
        title_container.configure(height=40)
        
        title_label = ttk.Label(title_container,
                              text=title,
                              style='FrameTitle.TLabel',
                              anchor='center')
        title_label.pack(fill="x", expand=True)
        
        # Create image container frame with gray background
        image_container = ttk.Frame(frame, style='Custom.TFrame')
        image_container.pack(expand=True, fill="both", padx=5, pady=5)
        image_container.configure(width=480, height=430)
        
        # Image label with gray background
        image_label = ttk.Label(image_container)
        image_label.pack(expand=True, fill="both")
        
        # Save button container
        save_container = ttk.Frame(frame, style='Custom.TFrame')
        save_container.pack(fill="x", pady=(5, 0))
        
        # Save button with enhanced style
        style.configure('Save.TButton',
                       font=('Helvetica', 10),
                       padding=5)
        
        save_btn = ttk.Button(save_container,
                            text="Save Image",
                            style='Save.TButton',
                            command=lambda: self.save_image(column))
        save_btn.pack(pady=5)
        
        if column == 0:
            self.original_image_label = image_label
        else:
            self.processed_image_label = image_label

    def display_image(self, image, label):
        """
        Display numpy array image in GUI label
        
        Args:
            image: numpy array of image data
            label: target label widget for display
        
        Maintains aspect ratio while fitting to display area
        """
        if image:
            # Use fixed dimensions for both frames to ensure they're identical
            frame_width = 480
            frame_height = 430
            
            # Calculate scaling ratio while preserving aspect ratio
            img_width, img_height = image.size
            width_ratio = frame_width / img_width
            height_ratio = frame_height / img_height
            scale_ratio = min(width_ratio, height_ratio)
            
            # Calculate new size
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with gray background
            final_image = Image.new('RGB', (frame_width, frame_height), '#f0f0f0')
            
            # Calculate position to center the image
            x_offset = (frame_width - new_width) // 2
            y_offset = (frame_height - new_height) // 2
            
            # Paste resized image onto background
            final_image.paste(resized_image, (x_offset, y_offset))
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(final_image)
            label.configure(image=photo)
            label.image = photo  # Keep reference

    def save_image(self, image_type):
        """Save the image to disk"""
        if image_type == 0 and self.current_image is not None:
            image_to_save = Image.fromarray(self.current_image)
            title = "Save Original Image"
        elif image_type == 1 and hasattr(self, 'processed_image'):
            image_to_save = Image.fromarray((self.processed_image * 255).astype(np.uint8))
            title = "Save Edge Detected Image"
        else:
            messagebox.showwarning("Warning", "No image to save!")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image_to_save.save(file_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def process_image_with_progress(self):
        """
        Execute edge detection pipeline with progress tracking
        
        Steps:
        1. Input validation
        2. Grayscale conversion
        3. Gaussian blur
        4. Gradient calculation
        5. Non-max suppression
        6. Double thresholding
        7. Hysteresis edge tracking
        """
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        self.progress_frame.grid()  # Show progress frame
        self.status_label.grid()  # Show status label
        
        try:
            # Convert to grayscale
            self.update_progress(20, "Converting to grayscale...")
            time.sleep(0.3)  # Simulate processing time
            gray_image = self.to_grayscale(self.current_image)
            
            # Apply Gaussian blur
            self.update_progress(40, "Applying Gaussian blur...")
            time.sleep(0.3)
            blurred = self.apply_gaussian_blur(gray_image)
            
            # Calculate gradients
            self.update_progress(60, "Calculating gradients...")
            time.sleep(0.3)
            gradient_magnitude, gradient_direction = self.sobel_filters(blurred)
            
            # Apply non-maximum suppression
            self.update_progress(70, "Applying non-maximum suppression...")
            time.sleep(0.3)
            suppressed = self.non_maximum_suppression(gradient_magnitude, gradient_direction)
            
            # Apply double threshold
            self.update_progress(80, "Applying double threshold...")
            time.sleep(0.3)
            strong_edges, weak_edges = self.double_threshold(suppressed)
            
            # Apply hysteresis
            self.update_progress(90, "Applying hysteresis...")
            time.sleep(0.3)
            final_edges = self.hysteresis(strong_edges, weak_edges)
            
            # Store processed image
            self.processed_image = final_edges
            
            # Display result
            self.update_progress(100, "Complete!")
            display_image = Image.fromarray((final_edges * 255).astype(np.uint8))
            self.display_image(display_image, self.processed_image_label)
            
            # Hide progress elements after a delay
            self.window.after(1000, self.hide_progress)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.hide_progress()

    def load_image(self):
        """
        Load an image through file dialog
        
        Supported formats: JPEG, PNG, BMP
        Updates original image display
        Handles common file errors
        """
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                image = Image.open(file_path)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Store original image for processing
                self.current_image = np.array(image)
                
                # Display image with proper scaling
                self.display_image(image, self.original_image_label)
                
                # Clear processed image
                if self.processed_image_label:
                    self.processed_image_label.configure(image='')
                
                self.status_label.config(text="Image loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_label.config(text="Failed to load image")

    def to_grayscale(self, image):
        """Convert RGB image to grayscale using manual implementation"""
        if len(image.shape) == 3:
            return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        return image

    def gaussian_kernel(self, size, sigma=1.4):
        """Generate Gaussian kernel manually"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for x in range(size):
            for y in range(size):
                x_dist = x - center
                y_dist = y - center
                kernel[x, y] = (1/(2*np.pi*sigma**2)) * np.exp(-(x_dist**2 + y_dist**2)/(2*sigma**2))
                
        return kernel / np.sum(kernel)

    def apply_gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian blur manually"""
        kernel = self.gaussian_kernel(kernel_size)
        padding = kernel_size // 2
        padded = np.pad(image, padding, mode='edge')
        output = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(
                    padded[i:i+kernel_size, j:j+kernel_size] * kernel
                )
        
        return output

    def sobel_filters(self, image):
        """Apply Sobel filters manually"""
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        padding = 1
        padded = np.pad(image, padding, mode='edge')
        gradient_x = np.zeros_like(image)
        gradient_y = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gradient_x[i, j] = np.sum(
                    padded[i:i+3, j:j+3] * Gx
                )
                gradient_y[i, j] = np.sum(
                    padded[i:i+3, j:j+3] * Gy
                )
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        return gradient_magnitude, gradient_direction

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        """Apply non-maximum suppression"""
        height, width = gradient_magnitude.shape
        output = np.zeros_like(gradient_magnitude)
        
        # Convert angles from radians to degrees
        angle = gradient_direction * 180 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                q = 255
                r = 255
                
                # Angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # Angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # Angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # Angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                    output[i,j] = gradient_magnitude[i,j]
                else:
                    output[i,j] = 0
                    
        return output

    def double_threshold(self, image, low_ratio=0.05, high_ratio=0.15):
        """Apply double threshold"""
        high_threshold = image.max() * high_ratio
        low_threshold = high_threshold * low_ratio
        
        strong_edges = (image >= high_threshold)
        weak_edges = (image >= low_threshold) & (image < high_threshold)
        
        return strong_edges, weak_edges

    def hysteresis(self, strong_edges, weak_edges):
        """Apply hysteresis to connect edges"""
        height, width = strong_edges.shape
        output = np.copy(strong_edges)
        
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        
        # Iterate until no more changes
        while True:
            previous = np.copy(output)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if weak_edges[i,j]:
                        # Check if any neighbor is a strong edge
                        for k in range(8):
                            if output[i + dx[k], j + dy[k]]:
                                output[i,j] = True
                                break
            
            if np.array_equal(previous, output):
                break
                
        return output
        
    def on_mousewheel(self, event):
        """Handle smooth mousewheel scrolling with improved animation"""
        # Get the delta value and normalize it
        delta = -1 * (event.delta // 120)
        
        # Use more steps with smaller increments for smoother animation
        steps = 15  # Increased number of steps for even smoother scrolling
        
        # Apply scrolling with acceleration and deceleration
        for i in range(steps):
            # Calculate a smooth deceleration curve
            factor = 1 - (i / steps) ** 2  # Quadratic deceleration for natural feel
            scroll_amount = max(1, int(delta * factor)) if delta > 0 else min(-1, int(delta * factor))
            # Apply with increasing delay for natural deceleration
            self.window.after(i * 4, lambda a=scroll_amount: self.main_canvas.yview_scroll(a, 'units'))

    def smooth_scroll(self, *args):
        """Implement smooth scrolling"""
        if len(args) > 1:
            self.main_canvas.yview_moveto(args[1])
        else:
            # Use smoother scrolling with acceleration
            amount = int(args[0])
            # Apply scrolling with acceleration effect
            if amount != 0:
                for i in range(5):  # Increased range for smoother scrolling
                    factor = 0.8 ** i  # Adjusted factor for better deceleration
                    scroll_amount = int(amount * factor) if amount * factor >= 1 or amount * factor <= -1 else amount
                    self.window.after(i * 4, lambda a=scroll_amount: self.main_canvas.yview_scroll(a, 'units'))

    def on_frame_configure(self, event=None):
        """Reset scroll region when content frame size changes"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        # Ensure the canvas is large enough to accommodate all content
        self.main_frame.update_idletasks()

    def on_canvas_configure(self, event):
        """Update canvas window size when canvas is resized"""
        self.main_canvas.itemconfig(self.canvas_frame, width=event.width)
        # Ensure the canvas window is properly sized for future content additions
        self.main_canvas.configure(width=event.width)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CannyEdgeDetector()
    app.run()