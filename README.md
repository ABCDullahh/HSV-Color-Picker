# HSV Color Picker
![image](https://github.com/user-attachments/assets/6e4108fb-14c9-49d3-ac25-0b594da9f8a1)

**HSV Color Picker** is an interactive GUI application, built with **PyQt5**, that empowers you to:

- Precisely **pick and adjust color values (H, S, V)** from either an opened image or directly from your screen.
- **Visualize masking** in real time based on HSV ranges you define.
- **Sample points** within your image and automatically tune HSV thresholds via **KMeans clustering**.
- Enjoy a **real-time Zoom Preview** as you move your cursor across the image.
- Apply **morphological operations (Erode & Dilate)** to polish your segmentation results.

## Key Features

1. **Open Image**  
   Load an image from local storage and preview it in the application.
2. **Pick Color**  
   Grab a color from anywhere on your screen by moving your mouse. The HSV sliders update automatically!
3. **Color Preview**  
   Visually confirm the hue, saturation, and value you’ve picked with handy color swatches.
4. **Range Control**  
   Define **LOWER** and **UPPER** HSV boundaries to isolate specific colors in your image.
5. **Auto Tune HSV**  
   Collect sample points from your image, then let the app figure out the best HSV thresholds using **KMeans**.
6. **Zoom Preview (Real-Time)**  
   Hover your cursor over the image to see an enlarged (8×) view of the area around your mouse position.
7. **Sampling Points**  
   Mark multiple points on your image and see crosshairs appear to keep track of them.
8. **Morphology**  
   Quickly toggle **Erode** or **Dilate** (with adjustable kernel size) to refine your mask and reduce noise.

## Getting Started

These instructions will help you set up and run the application on your local machine for development and testing purposes.

### Prerequisites

- **Python 3.7+** (earlier versions not tested)
- **pip** (to install dependencies)
  
### Installation

1. **Clone this repository** or download the source code (ZIP).
2. **(Optional)** Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install required packages**:
   ```bash
   pip install pyqt5 opencv-python numpy scikit-learn pyautogui
   pip install pywin32  # Windows-only, required for win32gui
   ```

### Running the Application

1. Navigate to the project folder in your terminal.
2. Run the main script:
   ```bash
   python hsv_picker.py
   ```
3. The application window will open. (It automatically scales to ~95% of your screen width and ~90% of your screen height.)

## Usage

1. **Open Image**  
   - Click “Open Image” and select a file. The loaded image displays in the right panel.  
2. **Pick Color**  
   - Click “Pick Color,” then hover your mouse over any area on your screen. The HSV sliders will update live.  
   - Click “Stop Picking” to end this mode.  
3. **Adjust HSV**  
   - Move the H, S, V sliders in the “HSV Controls” section.  
   - Switch between “LOWER” and “UPPER” from the drop-down to set each boundary.  
4. **View Mask & Masked Raw**  
   - The middle panel on the right shows the mask (white = included, black = excluded).  
   - The bottom panel shows the original image overlaid with that mask.  
5. **Sampling Points**  
   - Click “Sample Points,” then click on spots in your image to record their HSV values.  
   - Switch back to “Stop Sampling” once done.  
6. **Auto Tune HSV**  
   - After picking sample points, click “Auto Tune HSV.”  
   - The KMeans-based algorithm will suggest **Lower** and **Upper** HSV bounds automatically.  
7. **Morphology**  
   - Check “Erode” or “Dilate,” then adjust the sliders to remove noise or fill gaps.  
8. **Copy Range**  
   - Click “Copy Range” to copy the current HSV boundaries to your clipboard—ideal for quick code snippets.  

## Technical Details

- **Real-Time Zoom**: The left panel shows a magnified view of the area around your mouse pointer (8×).  
- **Mouse Tracking**: Enabled for the loaded image preview, so the zoom preview updates as you move the cursor.  
- **Clustering**: Uses **sklearn’s KMeans** for color sample grouping to find an optimal HSV range.  

## Contributing

1. **Fork** the project on GitHub.  
2. **Create your feature branch**: `git checkout -b feature/amazing-feature`.  
3. **Commit your changes**: `git commit -m 'Add some amazing feature'`.  
4. **Push to the branch**: `git push origin feature/amazing-feature`.  
5. Create a **Pull Request** to the main repository.  

Welcome any contributions to improve functionality, readability, or performance!


**Enjoy color picking and real-time masking with this handy tool!** If you have any questions or run into issues, feel free to open an [issue](https://github.com/ABCDullahh/HSV-Color-Picker/issues) or reach out. Have fun exploring your images in the HSV color space!
