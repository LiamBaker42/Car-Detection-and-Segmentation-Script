import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
import os 

# ==============================================================================
# CLASS 1: QUALITY MEASURES
# Evaluates how much the enhanced image mathematically differs from the original.
# ==============================================================================
class ImageQualityMetrics:
    "A class to calculate quality measures between an original and enhanced image."
    def __init__(self, original_img, enhanced_img):
        self.original = original_img
        self.enhanced = enhanced_img

    def calculate_mse(self):
        "Calculates Mean Squared Error. A lower score means less error/difference."
        err = np.sum((self.original.astype("float") - self.enhanced.astype("float")) ** 2)
        err /= float(self.original.shape[0] * self.original.shape[1] * self.original.shape[2])
        return err

    def calculate_psnr(self):
        "Calculates Peak Signal-to-Noise Ratio (dB). A higher score means better quality."
        mse = self.calculate_mse()
        if mse == 0:
            return float('inf') 
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def calculate_ssim(self):
        "Calculates Structural Similarity. A score of 1.0 means perfectly identical."
        score, _ = ssim(self.original, self.enhanced, full=True, channel_axis=2)
        return score

    def print_report(self):
        "Prints a clean, formatted report of all quality measures to the console."
        print("\n" + "="*40)
        print(" IMAGE QUALITY MEASURES REPORT")
        print("="*40)
        print(f"MSE  (Mean Squared Error):  {self.calculate_mse():.2f}  <-- (Lower is better)")
        print(f"PSNR (Peak Signal-to-Noise): {self.calculate_psnr():.2f} dB <-- (Higher is better)")
        print(f"SSIM (Structural Similarity): {self.calculate_ssim():.4f}  <-- (Closer to 1.0 is better)")
        print("="*40 + "\n")

# ==============================================================================
# VIDEO KEYFRAME EXTRACTOR
# ==============================================================================
def extract_keyframes_from_video(video_path, min_area=4000, cooldown_frames=30):
    "Scans a video for moving cars and extracts keyframes to pass to the pipeline."
    cap = cv2.VideoCapture(video_path)
    
    # MOG2 isolates moving objects from the static background
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    keyframes = []
    cooldown = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Apply subtraction to find motion
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        
        # Clean up motion noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # If recently captured a frame, wait for the car to leave the shot
        if cooldown > 0:
            cooldown -= 1
            continue
            
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            
            # If a massive moving object is detected
            if cv2.contourArea(best_cnt) > min_area:
                x, y, w, h = cv2.boundingRect(best_cnt)
                aspect = w / float(h)
                
                # Verify it roughly matches the proportions of a car
                if 1.0 < aspect < 5.0:
                    keyframes.append(frame.copy())
                    cooldown = cooldown_frames # Start cooldown to avoid capturing the same car
                    
    cap.release()
    return keyframes

# ==============================================================================
# FREQUENCY DOMAIN ENHANCER & SEGMENTATION
# Handles all image processing, car segmentation, part extraction, and matching.
# ==============================================================================
class FrequencyDomainEnhancer:
    def __init__(self, image_path=None, image_array=None, bg_image_path=None):
        # Load the Primary Image (From File OR directly from Video Array)
        if image_array is not None:
            self.img = image_array
        elif image_path is not None:
            self.img = cv2.imread(image_path)
            if self.img is None: raise ValueError(f"Could not load image at '{image_path}'")
        else:
            raise ValueError("You must provide either an image_path or an image_array.")
        
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.bg_img_rgb = None
        if bg_image_path:
            bg = cv2.imread(bg_image_path)
            if bg is not None: self.bg_img_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        self.dft = cv2.dft(np.float32(self.gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        self.dft_shift = np.fft.fftshift(self.dft)

        rows, cols = self.gray.shape
        crow, ccol = rows // 2, cols // 2
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        self.D = np.sqrt((U - ccol)**2 + (V - crow)**2) 

        self.mask_cleaned = None
        self.final_enhanced = None
        self.swapped_cars_img = None
        self.new_bg_composite = None

    # ---------------------------------------------------------
    # BACKGROUND ANALYSIS & MASK CREATION
    # ---------------------------------------------------------
    def _is_background_complex(self):
        "Analyzes the outer 10% border of the image to determine background complexity."
        h, w = self.gray.shape
        
        # Create a mask that only looks at the outer 10% edge of the picture
        border_mask = np.ones((h, w), dtype=np.uint8) * 255
        margin_y, margin_x = int(h * 0.1), int(w * 0.1)
        border_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 0
        
        # Check Variance (Are the border pixels all roughly the same color)
        border_pixels = self.gray[border_mask == 255]
        std_dev = np.std(border_pixels)
        
        # Check Edge Density (Are there lots of hard lines like buildings/trees in the background)
        edges = cv2.Canny(self.gray, 50, 150)
        border_edges = edges[border_mask == 255]
        edge_density = np.sum(border_edges > 0) / len(border_edges)
        
        # If the background is very smooth (low deviation) and has almost no lines, it's Plain.
        if std_dev < 45 and edge_density < 0.15:
            return False 
        return True
    
    # ---------------------------------------------------------
    # ORIENTATION DETECTION
    # ---------------------------------------------------------
    def _detect_car_orientation(self):
        "Uses Distance Transform 'Cores' to perfectly calculate aspect ratio, ignoring merges."
        h, w = self.img.shape[:2]
        blurred = cv2.GaussianBlur(self.img, (21, 21), 0)
        
        # Create a quick rough mask using top/bottom difference
        mean_top = np.mean(blurred[0:int(h*0.05), :], axis=(0,1))
        mean_bottom = np.mean(blurred[int(h*0.95):h, :], axis=(0,1))
        
        diff_top = cv2.absdiff(blurred, mean_top.astype(np.uint8))
        diff_bottom = cv2.absdiff(blurred, mean_bottom.astype(np.uint8))
        diff_combined = cv2.min(cv2.cvtColor(diff_top, cv2.COLOR_BGR2GRAY), 
                                cv2.cvtColor(diff_bottom, cv2.COLOR_BGR2GRAY))
        
        # Standard threshold so the cars become white silhouettes
        _, rough_mask = cv2.threshold(diff_combined, 80, 255, cv2.THRESH_BINARY)
        
        # Erase the top and bottom to ignore buildings/roads
        rough_mask[0:int(h*0.35), :] = 0
        rough_mask[int(h*0.90):h, :] = 0
        
        # Solidify the mask to fill in any windows or reflections
        rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        
        # ISOLATE THE CAR CORE
        # Instead of blindly eroding, find the topographical cores of the blobs.
        dist_transform = cv2.distanceTransform(rough_mask, cv2.DIST_L2, 5)
        
        # Isolate only the deepest 40% of the peaks
        _, cores = cv2.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
        cores = np.uint8(cores)
        
        # MEASURE THE GEOMETRY
        cnts, _ = cv2.findContours(cores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 'side-on' # Default fallback
        
        # Measure the aspect ratio of the largest perfectly isolated core
        largest_core = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_core)
        
        if ch == 0: return 'side-on'
        core_aspect_ratio = float(cw) / float(ch)
        
        # A side-on car's core is a long horizontal streak (Ratio usually > 2.0).  A head-on car's core is a square or vertical oval (Ratio usually < 1.3).
        if core_aspect_ratio < 1.6:
            return 'head-on'
        return 'side-on'

    # ---------------------------------------------------------
    # HEAD-ON COMPLEX MASKING
    # ---------------------------------------------------------
    def _create_headon_mask(self):
        """Optimized for head-on cars using Spatially-Aware Hints to kill buildings but save windshields."""
        h, w = self.img.shape[:2]
        gc_mask = np.zeros((h, w), np.uint8)
        gc_mask[:] = cv2.GC_PR_FGD  # Default everything to Probable Foreground
        
        blurred = cv2.GaussianBlur(self.img, (15, 15), 0)
        
        # ROAD HINT
        mean_bottom = np.mean(blurred[int(h*0.95):h, :], axis=(0,1))
        diff_bottom = cv2.absdiff(blurred, mean_bottom.astype(np.uint8))
        diff_bottom_gray = cv2.cvtColor(diff_bottom, cv2.COLOR_BGR2GRAY)
        
        gc_mask[diff_bottom_gray < 10] = cv2.GC_PR_BGD

        # SKY HINT
        # Learn the color of the building/sky from the top 5%
        mean_top = np.mean(blurred[0:int(h*0.05), :], axis=(0,1))
        diff_top = cv2.absdiff(blurred, mean_top.astype(np.uint8))
        diff_top_gray = cv2.cvtColor(diff_top, cv2.COLOR_BGR2GRAY)
        
        # Calculate the boundary for the top 40% of the image
        top_region_limit = int(h * 0.40)
        
        # Find all pixels that look like the building
        building_pixels = (diff_top_gray < 50)
        
        # only apply the Probable Background label to building pixels that are physically 
        # in the top 40% of the image. This isolates the building but completely ignores the windshield
        gc_mask[0:top_region_limit, :][building_pixels[0:top_region_limit, :]] = cv2.GC_PR_BGD

        # GRABCUT & WATERSHED
        # Define extreme edges as Definite Background (Slightly larger 4% top margin for head-on headroom)
        margin_y, margin_x = max(1, int(h * 0.04)), max(1, int(w * 0.02))
        gc_mask[0:margin_y, :] = cv2.GC_BGD
        gc_mask[h-max(1, int(h * 0.02)):h, :] = cv2.GC_BGD # Keep bottom margin thin
        gc_mask[:, 0:margin_x] = cv2.GC_BGD
        gc_mask[:, w-margin_x:w] = cv2.GC_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(self.img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        grabcut_mask = np.where((gc_mask==2)|(gc_mask==0), 0, 1).astype('uint8') * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_CLOSE, kernel)
        grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_OPEN, kernel)

        # Watershed Separation
        sure_bg = cv2.dilate(grabcut_mask, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(grabcut_mask, cv2.DIST_L2, 5)
        
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(self.img, markers)
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[markers > 1] = 255
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.mask_cleaned = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, smooth_kernel)
        
        # Final Cleanup: Keep only the Top 2 blobs
        cnts, _ = cv2.findContours(self.mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        clean_2_cars_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_2_cars_mask, cnts, -1, 255, -1)
        
        self.mask_cleaned = clean_2_cars_mask
        return self.mask_cleaned
    
    def _create_plain_mask(self, D0_low, open_ksize, close_ksize):
        "Original Frequency Domain logic for clean, solid backgrounds."
        self.D0_low = D0_low
        H_low = np.exp(-(self.D**2) / (2 * (self.D0_low**2)))
        fshift_low = self.dft_shift * np.dstack([H_low, H_low])
        img_back_low = cv2.idft(np.fft.ifftshift(fshift_low))
        img_smoothed = cv2.magnitude(img_back_low[:, :, 0], img_back_low[:, :, 1])

        img_smoothed = cv2.normalize(img_smoothed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, mask = cv2.threshold(img_smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_ksize), iterations=2)
        self.mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ksize), iterations=3)
        return self.mask_cleaned

    def _create_complex_mask(self):
        "Uses Top/Bottom Difference to feed Hints to GrabCut, Watershed, and Contour Filtering."
        h, w = self.img.shape[:2]
        
        # GENERATE HINTS FOR GRABCUT
        gc_mask = np.zeros((h, w), np.uint8)
        gc_mask[:] = cv2.GC_PR_FGD  # Default everything to Probable Foreground
        
        # Sample the top 5% (sky/building) and bottom 5% (road)
        blurred = cv2.GaussianBlur(self.img, (15, 15), 0)
        mean_top = np.mean(blurred[0:int(h*0.05), :], axis=(0,1))
        mean_bottom = np.mean(blurred[int(h*0.95):h, :], axis=(0,1))
        
        # Find how different every pixel is from the top or bottom
        diff_top = cv2.absdiff(blurred, mean_top.astype(np.uint8))
        diff_bottom = cv2.absdiff(blurred, mean_bottom.astype(np.uint8))
        
        diff_top_gray = cv2.cvtColor(diff_top, cv2.COLOR_BGR2GRAY)
        diff_bottom_gray = cv2.cvtColor(diff_bottom, cv2.COLOR_BGR2GRAY)
        diff_combined = cv2.min(diff_top_gray, diff_bottom_gray)
        
        gc_mask[diff_combined < 70] = cv2.GC_PR_BGD

        # DEFINITE BACKGROUND & GRABCUT EXECUTION
        # Define the extreme 2% edge as Definite Background to anchor the algorithm
        margin_y, margin_x = max(1, int(h * 0.02)), max(1, int(w * 0.02))
        gc_mask[0:margin_y, :] = cv2.GC_BGD
        gc_mask[h-margin_y:h, :] = cv2.GC_BGD
        gc_mask[:, 0:margin_x] = cv2.GC_BGD
        gc_mask[:, w-margin_x:w] = cv2.GC_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Run GrabCut
        cv2.grabCut(self.img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # Extract the binary mask
        grabcut_mask = np.where((gc_mask==2)|(gc_mask==0), 0, 1).astype('uint8') * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_CLOSE, kernel)
        grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_OPEN, kernel)

        # Slice cars apart if they merged
        sure_bg = cv2.dilate(grabcut_mask, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(grabcut_mask, cv2.DIST_L2, 5)
        
        # Find the deep centers (the core of each car)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply Watershed Algorithm
        markers = cv2.watershed(self.img, markers)
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[markers > 1] = 255
        
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.mask_cleaned = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, smooth_kernel)
        
        # Keep ONLY the 2 largest blobs
        cnts, _ = cv2.findContours(self.mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort them by Area (Largest to smallest) and keep only the top 2
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        
        # Create a completely blank canvas and draw only those two massive car blobs onto it
        clean_2_cars_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_2_cars_mask, cnts, -1, 255, -1)
        
        self.mask_cleaned = clean_2_cars_mask
        
        return self.mask_cleaned

    def create_mask(self, D0_low=15, open_ksize=(9, 9), close_ksize=(36, 36)):
        "Dispatcher that analyzes background complexity and car orientation to pick the best mask."
        if self._is_background_complex():
            orientation = self._detect_car_orientation()
            if orientation == 'head-on':
                print("--> Detected COMPLEX background & HEAD-ON cars. Using Head-On Masking.")
                return self._create_headon_mask()
            else:
                print("--> Detected COMPLEX background & SIDE-ON cars. Using Side-Profile Masking.")
                return self._create_complex_mask()
        else:
            print("--> Detected PLAIN background. Using Frequency Domain Masking.")
            return self._create_plain_mask(D0_low, open_ksize, close_ksize)
        
    # ---------------------------------------------------------
    # IMAGE ENHANCEMENT: Fusion of Spatial and Frequency Domains
    # ---------------------------------------------------------
    def enhance_cars(self, D0_high=30, sharp_strength=0.5, clahe_clip=1.2):
        "Fuses Spatial Domain (Lighting fixes) with Frequency Domain (Edge sharpening)."
        if self.mask_cleaned is None: raise RuntimeError("Call create_mask() first.")
        self.D0_high = D0_high
        
        # SPATIAL DOMAIN: Fix Lighting & Contrast
        # Convert to LAB space so only adjust lightness, not color
        lab = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to pull details out of the shadows
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        limg = cv2.merge((cl, a_channel, b_channel))
        spatial_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Apply Bilateral Filter to smooth out noisy paint while keeping edges sharp
        spatial_smoothed = cv2.bilateralFilter(spatial_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # FREQUENCY DOMAIN: Extract Sharp Edges 
        # Apply Gaussian Highpass Filter (blocks general shapes, keeps sharp edges)
        H_high = 1 - np.exp(-(self.D**2) / (2 * (self.D0_high**2)))
        H_high_2d = np.dstack([H_high, H_high])
        fshift_high = self.dft_shift * H_high_2d
        
        f_ishift_high = np.fft.ifftshift(fshift_high)
        img_back_high = cv2.idft(f_ishift_high)
        highpass_edges = cv2.magnitude(img_back_high[:, :, 0], img_back_high[:, :, 1])

        highpass_edges = cv2.normalize(highpass_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        edges_3c = cv2.cvtColor(highpass_edges, cv2.COLOR_GRAY2RGB)
        
        # FUSION: Combine Both Enhancements
        # Blend the sharp edges directly onto the spatially-lit image
        fused_img = cv2.addWeighted(spatial_smoothed, 1.0, edges_3c, sharp_strength, 0)

        # SAVE INTERMEDIATES FOR PLOTTING
        self.spatial_component = spatial_smoothed
        self.frequency_component = edges_3c
        self.fused_full = fused_img

        # Apply the final enhancement ONLY to the cars, leaving the background original
        enhanced_cars = cv2.bitwise_and(fused_img, fused_img, mask=self.mask_cleaned)
        inv_mask = cv2.bitwise_not(self.mask_cleaned)
        original_background = cv2.bitwise_and(self.img_rgb, self.img_rgb, mask=inv_mask)
        
        self.final_enhanced = cv2.add(enhanced_cars, original_background)
        return self.final_enhanced


    # ---------------------------------------------------------
    # PART IDENTIFICATION: Finds Wheels, Windows, and Lights
    # ---------------------------------------------------------
    def create_circle_contour(self, center, radius):
        "Helper function: Generates a perfect circular contour for wheels."
        theta = np.linspace(0, 2 * np.pi, 50)
        x_pts = center[0] + radius * np.cos(theta)
        y_pts = center[1] + radius * np.sin(theta)
        cnt = np.column_stack([x_pts, y_pts]).astype(np.int32)
        return cnt.reshape((-1, 1, 2))

    def validate_spatial_relationship(self, part_center, hull):
        "Helper function: Ensures a detected part is actually inside the car's body."
        dist = cv2.pointPolygonTest(hull, part_center, False)
        return dist >= 0

    def _identify_parts(self, img, mask, rect):
        "Pre-processes complex images to look like simple high-contrast ones."
        x, y, w, h = rect
        roi = img[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        # ZONAL SANITIZATION STEP
        cabin_limit = int(h * 0.5)   # Top 50% for Windows/Lights
        wheel_limit = int(h * 0.6)   # Bottom 30% for Wheels

        # Process Cabin (Heavy Smoothing to remove reflections)
        cabin_roi = roi[:cabin_limit, :]
        sanitized_cabin = cv2.bilateralFilter(cabin_roi, d=15, sigmaColor=100, sigmaSpace=100)
        sanitized_cabin = cv2.GaussianBlur(sanitized_cabin, (5, 5), 0)
        wheel_roi = roi[wheel_limit:, :]
        
        # Convert to grayscale for contrast stretching
        wheel_gray = cv2.cvtColor(wheel_roi, cv2.COLOR_RGB2GRAY)
        
        # extracts bright features (rims) and dark features (tires) from the background
        kernel_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blackhat = cv2.morphologyEx(wheel_gray, cv2.MORPH_BLACKHAT, kernel_th)
        
        # Combine the original with the blackhat to boost the dark tire edges
        wheel_enhanced = cv2.add(wheel_gray, blackhat)
        
        # Apply a sharp Bilateral filter that preserves edges but kills asphalt texture
        sanitized_wheel = cv2.bilateralFilter(cv2.cvtColor(wheel_enhanced, cv2.COLOR_GRAY2RGB), 
                                              d=7, sigmaColor=30, sigmaSpace=30)

        # 4. Reconstruct Sanitized ROI from zones
        roi_clean = roi.copy()
        roi_clean[:cabin_limit, :] = sanitized_cabin
        roi_clean[wheel_limit:, :] = sanitized_wheel

        # 5. Flatten Background
        roi_clean = cv2.bitwise_and(roi_clean, roi_clean, mask=roi_mask)

        parts = []
        H, W = roi_clean.shape[:2]
        gray = cv2.cvtColor(roi_clean, cv2.COLOR_RGB2GRAY)
        
        # Determine Orientation for THIS specific car
        aspect_ratio = float(W) / float(H)
        orientation = 'head-on' if aspect_ratio < 1.6 else 'side-on'
        
        # 1. CHASSIS: (Applies to both orientations)
        body_cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chassis_hull = None
        if body_cnts:
            main_body_cnt = max(body_cnts, key=cv2.contourArea)
            chassis_hull = cv2.convexHull(main_body_cnt)
            parts.append({
                'label': 'Chassis', 'img': roi.copy(),
                'rel_x': 0, 'rel_y': 0, 'w': W, 'h': H,
                'contour': main_body_cnt, 'hull': chassis_hull, 'type': 'Parent'
            })

        # =====================================================================
        # SIDE-ON LOGIC
        # =====================================================================
        if orientation == 'side-on':
            # 2. WHEELS (Side-On): Look for perfect circles and mask them perfectly
            search_strip_y = int(H*0.55)
            bottom = cv2.GaussianBlur(gray[search_strip_y:, :], (9, 9), 4)
            min_r, max_r = max(10, int(H * 0.12)), int(H * 0.28) 
            circles = cv2.HoughCircles(bottom, cv2.HOUGH_GRADIENT, dp=1.4, minDist=W//5, param1=50, param2=40, minRadius=min_r, maxRadius=max_r)
            self.debug_orientation = "SideOn"

            wheel_found = False
            if circles is not None:
                for (cx, cy, r) in np.round(circles[0, :]).astype("int"):
                    true_cy = search_strip_y + cy
                    if chassis_hull is not None and not self.validate_spatial_relationship((float(cx), float(true_cy)), chassis_hull): continue
                    
                    # Draw the detection on the debug mask for Grid #10
                    cv2.circle(self.debug_tire_thresh, (x + cx, y + true_cy), r, 255, -1)
                    x1, x2, y1, y2 = max(0, cx-r), min(W, cx+r), max(0, true_cy-r), min(H, true_cy+r)
                    
                    p_img_rect = roi[y1:y2, x1:x2]
                    
                    if p_img_rect.size > 0:
                        # Create a blank black mask matching the size of square cutout
                        circle_mask = np.zeros(p_img_rect.shape[:2], dtype=np.uint8)
                        # Find the center of the wheel relative to the small square just cut
                        rel_cx, rel_cy = cx - x1, true_cy - y1
                        # Draw a solid white circle on the mask
                        cv2.circle(circle_mask, (rel_cx, rel_cy), r, 255, -1)
                        # Apply the mask: Inside the circle stays visible, outside goes black
                        p_img_circular = cv2.bitwise_and(p_img_rect, p_img_rect, mask=circle_mask)
                        
                        parts.append({'label': 'Wheel', 'img': p_img_circular.copy(), 'rel_x': x1, 'rel_y': y1, 'w': x2-x1, 'h': y2-y1, 'contour': self.create_circle_contour((cx, true_cy), r), 'type': 'Child'})
                        wheel_found = True

            # Fallback: If Hough fails, force the dark tire contours into perfect circles too
            if not wheel_found: 
                fallback_y = int(H*0.3)
                _, tire_thresh = cv2.threshold(gray[fallback_y:, :], 60, 255, cv2.THRESH_BINARY_INV)
                self.debug_tire_thresh[y+fallback_y:y+H, x:x+W] = tire_thresh
                for c in cv2.findContours(cv2.bitwise_and(tire_thresh, tire_thresh, mask=roi_mask[fallback_y:, :]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    tx, ty, tw, th = cv2.boundingRect(c)
                    center_x = tx + tw//2
                    
                    if cv2.contourArea(c) > 200 and ((center_x < W * 0.25) or (center_x > W * 0.75)) and (0.3 < tw/float(th) < 1.5):
                        # Force the irregular blob into a perfect circle mathematically
                        (circle_x, circle_y), circle_r = cv2.minEnclosingCircle(c)
                        circle_x, circle_y, circle_r = int(circle_x), int(circle_y) + fallback_y, int(circle_r)
                        
                        x1, x2 = max(0, circle_x - circle_r), min(W, circle_x + circle_r)
                        y1, y2 = max(0, circle_y - circle_r), min(H, circle_y + circle_r)
   
                        p_img_rect = roi[y1:y2, x1:x2]
                        
                        if p_img_rect.size > 0:
                            circle_mask = np.zeros(p_img_rect.shape[:2], dtype=np.uint8)
                            rel_cx, rel_cy = circle_x - x1, circle_y - y1
                            cv2.circle(circle_mask, (rel_cx, rel_cy), circle_r, 255, -1)
                            p_img_circular = cv2.bitwise_and(p_img_rect, p_img_rect, mask=circle_mask)
                            
                            c_shifted = self.create_circle_contour((circle_x, circle_y), circle_r)
                            parts.append({'label': 'Wheel', 'img': p_img_circular.copy(), 'rel_x': x1, 'rel_y': y1, 'w': x2-x1, 'h': y2-y1, 'contour': c_shifted, 'type': 'Child'})

            # 3. WINDOWS (Side-On)
            top_h = int(H * 0.25)
            top_crop = roi_clean[:top_h, :] 
            top_gray = cv2.cvtColor(top_crop, cv2.COLOR_RGB2GRAY)
            
            # --- FEATURE EXTRACTION: LBP + RGB ---
            # LBP helps distinguish the smooth glass texture from the textured car body
            lbp = local_binary_pattern(top_gray, 8, 1, method='uniform')
            lbp_normalized = np.float32(lbp) * (255.0 / (lbp.max() or 1))
            pixel_values_rgb = np.float32(top_crop.reshape((-1, 3)))
            combined_features = np.hstack((pixel_values_rgb, np.float32(lbp_normalized.reshape((-1, 1)))))
            non_black_indices = np.where(np.any(pixel_values_rgb != [0, 0, 0], axis=1))[0]
            
            if len(non_black_indices) > 0:
                # Cluster pixels into 2 groups: Car Body and Windows/Details
                _, labels, centers = cv2.kmeans(combined_features[non_black_indices], 2, None, 
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
                                                10, cv2.KMEANS_RANDOM_CENTERS)
                
                # 1. Create the raw mask
                win_thresh_raw = np.zeros(len(combined_features), dtype=np.uint8)
                win_thresh_raw[non_black_indices[labels.flatten() == np.argmin(np.sum(centers[:, :3], axis=1))]] = 255
                win_thresh_raw = win_thresh_raw.reshape((top_h, W))

                # 2. Strong Horizontal Bridge (Connects front and back windows)
                # This acts like a rubber band stretched horizontally
                horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                win_thresh_stable = cv2.morphologyEx(win_thresh_raw, cv2.MORPH_CLOSE, horiz_kernel)

                # 3. Clean up the edges to stay inside the car body
                win_thresh_stable = cv2.bitwise_and(win_thresh_stable, win_thresh_stable, mask=roi_mask[:top_h, :])

                # 4. Invert to get the glass
                inverted_windows = cv2.bitwise_not(win_thresh_stable)
                just_glass = cv2.bitwise_and(inverted_windows, inverted_windows, mask=roi_mask[:top_h, :])

                # 5. Final structural cleanup
                just_glass = cv2.morphologyEx(just_glass, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
                                

                self.debug_win_thresh[y:y+top_h, x:x+W] = just_glass
                window_cnts, _ = cv2.findContours(just_glass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                window_cnts = sorted(window_cnts, key=lambda c: cv2.boundingRect(c)[0])
                
                for c in window_cnts:
                    area = cv2.contourArea(c)
                    if area > 200: 
                        wx, wy, ww, wh = cv2.boundingRect(c)
                        aspect_ratio = ww / float(wh)
                        
                        if 5 > aspect_ratio > 1.1:
                            if chassis_hull is not None:
                                if not self.validate_spatial_relationship((float(wx + ww//2), float(wy + wh//2)), chassis_hull):
                                    continue
                            parts.append({
                                'label': 'Window', 
                                'img': roi[wy:wy+wh, wx:wx+ww].copy(), 
                                'rel_x': wx, 'rel_y': wy, 'w': ww, 'h': wh, 
                                'contour': cv2.convexHull(c), 
                                'type': 'Child'
                            })

            # 4. LIGHTS (Side-On)
            l_h_start, l_h_end = int(H * 0.25), int(H * 0.55)
            l_crop = roi_clean[l_h_start:l_h_end, :]
            l_mask = roi_mask[l_h_start:l_h_end, :]

            # Sample the middle 40% of the car body to get the paint color
            mid_start, mid_end = int(W * 0.3), int(W * 0.7)
            paint_sample = l_crop[:, mid_start:mid_end]
            paint_mask = l_mask[:, mid_start:mid_end]
            
            # Get the average RGB color of the paint, ignoring the black background
            if cv2.countNonZero(paint_mask) > 0:
                paint_color = cv2.mean(paint_sample, mask=paint_mask)[:3]
            else:
                paint_color = (127, 127, 127)
                
            # Create a blank image entirely filled with the car's paint color
            paint_block = np.zeros_like(l_crop)
            paint_block[:] = paint_color
            
            # Find the absolute difference between the actual image and the pure paint color
            color_diff = cv2.absdiff(l_crop, paint_block)
            diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_RGB2GRAY)
            
            # Any pixel that is highly different from the paint color turns white
            _, non_paint_mask = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
            
            # Clamp it so it strictly stays inside the physical car body silhouette
            non_paint_mask = cv2.bitwise_and(non_paint_mask, l_mask)

            # Only look at the front 10% and back 10%
            l_final_mask = np.zeros_like(non_paint_mask)
            front_zone = int(W * 0.10)
            back_zone = int(W * 0.90)
            
            l_final_mask[:, :front_zone] = non_paint_mask[:, :front_zone]
            l_final_mask[:, back_zone:] = non_paint_mask[:, back_zone:]

            # Minor Cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            l_final_mask = cv2.morphologyEx(l_final_mask, cv2.MORPH_CLOSE, kernel)
            l_final_mask = cv2.morphologyEx(l_final_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            self.debug_dog_lights[y+l_h_start:y+l_h_end, x:x+W] = l_final_mask

            l_cnts, _ = cv2.findContours(l_final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in l_cnts:
                lx, ly, lw, lh = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                
                if 10 < area < (W * H * 0.1):
                    global_ly = l_h_start + ly
                    
                    # SHIFT THE CONTOUR POINTS 
                    c_shifted = c.copy()
                    c_shifted[:, :, 1] += l_h_start 
                    
                    part_roi = roi[global_ly:global_ly+lh, lx:lx+lw].copy()
                    part_mask = l_final_mask[ly:ly+lh, lx:lx+lw]
                    masked_light = cv2.bitwise_and(part_roi, part_roi, mask=part_mask)

                    parts.append({
                        'label': 'Light', 
                        'img': masked_light, 
                        'rel_x': lx, 'rel_y': global_ly, 
                        'w': lw, 'h': lh, 
                        'contour': c_shifted, 
                        'type': 'Child'
                    })
        # =====================================================================
        # HEAD-ON LOGIC (Strict Spatial Clamping)
        # =====================================================================
        else:
            # 1. WINDOWS (Head-On): The Windshield
            top_h = int(H * 0.35)
            top_gray = gray[:top_h, :]
            top_mask = roi_mask[:top_h, :]
            self.debug_orientation = "HeadOn"
            mean_val = cv2.mean(top_gray, mask=top_mask)[0]
            _, win_thresh = cv2.threshold(top_gray, int(mean_val * 0.9), 255, cv2.THRESH_BINARY_INV)
            win_thresh = cv2.bitwise_and(win_thresh, top_mask)
            
            # Windshields are extremely wide. Use a massive horizontal kernel to force fragments to touch.
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 9))
            win_thresh = cv2.morphologyEx(win_thresh, cv2.MORPH_CLOSE, bridge_kernel)

            solid_win_mask = np.zeros_like(win_thresh)
            cnts, _ = cv2.findContours(win_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                # Windshields span almost the whole width of the car, so grab the widest shape
                widest_win = max(cnts, key=lambda c: cv2.boundingRect(c)[2])
                wx, wy, ww, wh = cv2.boundingRect(widest_win)
                
                if ww > (W * 0.4) and ww > wh:
                    hull = cv2.convexHull(widest_win)
                    cv2.drawContours(solid_win_mask, [hull], -1, 255, -1)
                    
                    # Send this perfectly solid mask to Grid 9
                    self.debug_win_thresh[y:y+top_h, x:x+W] = solid_win_mask

                    part_roi = roi[wy:wy+wh, wx:wx+ww].copy()
                    part_mask = solid_win_mask[wy:wy+wh, wx:wx+ww]
                    masked_window = cv2.bitwise_and(part_roi, part_roi, mask=part_mask)

                    parts.append({
                        'label': 'Window', 
                        'img': masked_window, 
                        'rel_x': wx, 'rel_y': wy, 'w': ww, 'h': wh, 
                        'contour': hull, 'type': 'Child'
                    })

            # 2. HEADLIGHTS (Refined Saturation & Brightness Filtering)
            belt_top, belt_bottom = int(H * 0.35), int(H * 0.65)
            belt_roi = roi[belt_top:belt_bottom, :]
            belt_mask = roi_mask[belt_top:belt_bottom, :]

            # Convert to HSV
            hsv_belt = cv2.cvtColor(belt_roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_belt)

            # Headlights are bright OR very desaturated (chrome/glass)
            bright_mask = cv2.threshold(v, 210, 255, cv2.THRESH_BINARY)[1] 
            desaturated_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)[1] 
            not_dark_mask = cv2.threshold(v, 110, 255, cv2.THRESH_BINARY)[1] 

            # Combine masks
            light_binary = cv2.bitwise_or(bright_mask, desaturated_mask)
            light_binary = cv2.bitwise_and(light_binary, not_dark_mask)
            light_binary = cv2.bitwise_and(light_binary, belt_mask)

            # Morphological Opening removes small pieces and noise while keeping large objects
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            light_binary = cv2.erode(light_binary, kernel_clean, iterations=2) 
            light_binary = cv2.dilate(light_binary, kernel_clean, iterations=1)

            # Fuse the remaining pieces
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            light_binary = cv2.morphologyEx(light_binary, cv2.MORPH_CLOSE, kernel_close)

            # Turn into solid shapes
            solid_light_mask = np.zeros_like(light_binary)
            cnts, _ = cv2.findContours(light_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                lx, ly, lw, lh = cv2.boundingRect(c)
                center_x = lx + (lw // 2)
                aspect_ratio = float(lw) / lh
                
                #Restrict to outer 30% and filter by aspect ratio 
                if 150 < area < (W * H * 0.20): 
                    if 0.8 < aspect_ratio < 4.0: 
                        if center_x < W * 0.28 or center_x > W * 0.72: 
                            
                            hull = cv2.convexHull(c)
                            cv2.drawContours(solid_light_mask, [hull], -1, 255, -1)
                            
                            global_ly = belt_top + ly
                            c_shifted = hull.copy()
                            c_shifted[:, :, 1] += belt_top
                            
                            parts.append({
                                'label': 'Light', 
                                'img': roi[global_ly:global_ly+lh, lx:lx+lw].copy(), 
                                'rel_x': lx, 'rel_y': global_ly, 'w': lw, 'h': lh, 
                                'contour': c_shifted, 'type': 'Child'
                            })

            # Update the debug grid with the SOLID masks, not the edges
            self.debug_dog_lights[y+belt_top:y+belt_bottom, x:x+W] = solid_light_mask

            # 3. WHEELS (Head-On): The Bottom Corners
            bottom_y = int(H * 0.7)
            bottom_gray = gray[bottom_y:, :]
            bottom_mask = roi_mask[bottom_y:, :]
            
            # Use Otsu to find the absolute darkest pixels (the tire rubber/shadows)
            _, tire_thresh = cv2.threshold(bottom_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            tire_thresh = cv2.bitwise_and(tire_thresh, bottom_mask)
            
            # Use a VERTICAL kernel to stretch the blobs up and down to break them from floor shadows
            vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
            tire_thresh = cv2.morphologyEx(tire_thresh, cv2.MORPH_OPEN, vert_kernel)

            solid_tire_mask = np.zeros_like(tire_thresh)
            
            for c in cv2.findContours(tire_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                tx, ty, tw, th = cv2.boundingRect(c)
                center_x, global_ty = tx + tw//2, bottom_y + ty
                
                # Force it to only look at the extreme outer 20% corners for the tires
                if cv2.contourArea(c) > 50 and (center_x < W * 0.20 or center_x > W * 0.80):
                    # Wrap the detected tire pixels in a convex hull for a clean, solid shape
                    hull = cv2.convexHull(c)
                    cv2.drawContours(solid_tire_mask, [hull], -1, 255, -1)
                    
                    c_shifted = hull.copy()
                    c_shifted[:, :, 1] += bottom_y
                    
                    parts.append({
                        'label': 'Wheel', 
                        'img': roi[global_ty:global_ty+th, tx:tx+tw].copy(), 
                        'rel_x': tx, 'rel_y': global_ty, 'w': tw, 'h': th, 
                        'contour': c_shifted, 'type': 'Child'
                    })

            # Pass the SOLID smoothed mask to the final display grid (Grid 10)
            self.debug_tire_thresh[y+bottom_y:y+H, x:x+W] = solid_tire_mask
         
        # 5. SUBTRACTIVE LOGIC (Applies to both)
        chassis_part = next((p for p in parts if p['label'] == 'Chassis'), None)
        if chassis_part:
            for p in parts:
                if p['type'] == 'Child':
                    # Use the part's actual contour/hull for a precise punch out
                    if p['contour'] is not None:
                        # Draw the specific shape of the part in black onto the chassis
                        cv2.drawContours(chassis_part['img'], [p['contour']], -1, (0, 0, 0), -1)
                    else:
                        # Fallback to rectangle only if contour is missing
                        cv2.rectangle(chassis_part['img'], (p['rel_x'], p['rel_y']), 
                                    (p['rel_x']+p['w'], p['rel_y']+p['h']), (0, 0, 0), -1)

        return roi_clean, parts


    # ---------------------------------------------------------
    # IMAGE RECONSTRUCTION & COMPOSITING
    # ---------------------------------------------------------
    def reconstruct_car(self, parts, w, h, template=None):
        """
        Rebuilds the car by shape-matching extracted parts to the black holes.
        Ensures the search area is always larger than the part to avoid OpenCV errors.
        """
        chassis_part = next((p for p in parts if p['label'] == 'Chassis'), None)
        canvas = chassis_part['img'].copy() if chassis_part is not None else template.copy()

        for p in [part for part in parts if part['type'] == 'Child']:
            part_img = p['img']
            ph, pw = part_img.shape[:2]
            label = p['label']
            
            # DEFINE INITIAL SEARCH ZONES
            y_min, y_max = (0, h)
            x_min, x_max = (0, w)
            aspect_ratio = float(w) / float(h)
            is_head_on = aspect_ratio < 1.6
            
            if label == 'Window':
                if is_head_on:
                    y_min, y_max = 0, int(h * 0.45)
                else:
                    y_min, y_max = 0, int(h * 0.3)
                    if p['rel_x'] < w * 0.5: x_max = int(w * 0.6)
                    else: x_min = int(w * 0.5)
            elif label == 'Light':
                if is_head_on:
                    y_min, y_max = int(h * 0.25), int(h * 0.65)
                    if p['rel_x'] < w * 0.5: x_max = int(w * 0.5)
                    else: x_min = int(w * 0.5)
                else:
                    y_min, y_max = int(h * 0.25), int(h * 0.65)
                    if p['rel_x'] < w * 0.5: x_max = int(w * 0.35)
                    else: x_min = int(w * 0.65)
            elif label == 'Wheel':
                y_min, y_max = int(h * 0.4), h 
                if p['rel_x'] < w * 0.5: x_max = int(w * 0.5)
                else: x_min = int(w * 0.5)

            # CRITICAL SAFETY EXPANSION
            if (y_max - y_min) < ph:
                y_max = min(h, y_min + ph + 2)
                if (y_max - y_min) < ph: 
                    y_min = max(0, y_max - ph - 2)

            if (x_max - x_min) < pw:
                x_max = min(w, x_min + pw + 2)
                if (x_max - x_min) < pw: 
                    x_min = max(0, x_max - pw - 2)

            # Slice the search zone from the canvas
            search_zone = canvas[y_min:y_max, x_min:x_max]

            # SHAPE-BASED MATCHING
            hole_mask = cv2.inRange(search_zone, np.array([0, 0, 0]), np.array([0, 0, 0]))
            
            # Generate the Part mask
            gray_p = cv2.cvtColor(part_img, cv2.COLOR_RGB2GRAY)
            _, p_mask = cv2.threshold(gray_p, 1, 255, cv2.THRESH_BINARY)
            
            if label == 'Wheel':
                p_mask = np.zeros((ph, pw), dtype=np.uint8)
                cv2.circle(p_mask, (pw//2, ph//2), int(min(pw, ph)*0.48), 255, -1)

            # Match the shape of the part to the shape of the hole
            res = cv2.matchTemplate(hole_mask, p_mask, cv2.TM_CCORR_NORMED)
            
            _, _, _, max_loc = cv2.minMaxLoc(res)
            fit_x, fit_y = x_min + max_loc[0], y_min + max_loc[1]

            # FINAL PLACEMENT
            if fit_y + ph <= canvas.shape[0] and fit_x + pw <= canvas.shape[1]:
                roi = canvas[fit_y:fit_y+ph, fit_x:fit_x+pw]
                img_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(p_mask))
                img_fg = cv2.bitwise_and(part_img, part_img, mask=p_mask)
                canvas[fit_y:fit_y+ph, fit_x:fit_x+pw] = cv2.add(img_bg, img_fg)
            
        return canvas

    def apply_new_background(self):
        "Places the newly segmented cars onto the provided background image."
        if self.bg_img_rgb is None:
            self.new_bg_composite = self.img_rgb.copy()
            return

        bg_resized = cv2.resize(self.bg_img_rgb, (self.img_rgb.shape[1], self.img_rgb.shape[0]))
        
        fg_cars = cv2.bitwise_and(self.img_rgb, self.img_rgb, mask=self.mask_cleaned)
        bg_masked = cv2.bitwise_and(bg_resized, bg_resized, mask=cv2.bitwise_not(self.mask_cleaned))
        
        self.new_bg_composite = cv2.add(fg_cars, bg_masked)

    def create_swapped_image(self, all_parts, valid_contours):
        "Swaps parts by matching the SHAPE of donor parts to the SHAPE of the target holes. Ignores texture/paint to ensure a perfect fit."
        self.swapped_cars_img = self.img_rgb.copy()
        if len(all_parts) < 2: return

        # Iterate over both cars. Target = Car receiving parts. Source = Car donating parts.
        for target_idx, source_idx in [(0, 1), (1, 0)]:
            target_parts = all_parts[target_idx]
            source_parts = all_parts[source_idx]
        
            x, y, w, h = cv2.boundingRect(valid_contours[target_idx])
            
            # search directly on the chassis of the target car
            chassis_part = next((p for p in target_parts if p['label'] == 'Chassis'), None)
            if not chassis_part: continue
            canvas = chassis_part['img'].copy()
            
            for label in ['Wheel', 'Window', 'Light']:
                t_items = [p for p in target_parts if p['label'] == label]
                s_items = [p for p in source_parts if p['label'] == label]
                
                for t_part, s_part in zip(t_items, s_items):
                    # Define a focused search zone around the original location
                    margin = int(w * 0.15) 
                    x_min = max(0, t_part['rel_x'] - margin)
                    x_max = min(w, t_part['rel_x'] + t_part['w'] + margin)

                    s_img = s_part['img'].copy()
                    th, tw = t_part['img'].shape[:2] 
                    s_img = cv2.resize(s_img, (tw, th))
                    
                    # Apply mirroring if needed
                    aspect_ratio = float(w) / float(h)
                    is_head_on = aspect_ratio < 1.6
                    if label == 'Window' and is_head_on: 
                        s_img = cv2.flip(s_img, 1)

                    # Generate the Shape Mask for the donor part
                    gray_s = cv2.cvtColor(s_img, cv2.COLOR_RGB2GRAY)
                    _, s_mask = cv2.threshold(gray_s, 1, 255, cv2.THRESH_BINARY)
                    
                    if label == 'Wheel':
                        s_mask = np.zeros((th, tw), dtype=np.uint8)
                        cv2.circle(s_mask, (tw//2, th//2), int(min(tw, th)*0.48), 255, -1)

                    # DEFINE THE SEARCH ZONE
                    y_min, y_max = (0, h)
                    x_min, x_max = (0, w)
                    
                    if label == 'Window': 
                        if is_head_on:
                            y_min, y_max = 0, int(h * (0.45))
                        else:
                            y_min, y_max = 0, int(h * (0.5))
                            if t_part['rel_x'] < w * 0.5: x_max = int(w * 0.6)
                            else: x_min = int(w * 0.5)
                    elif label == 'Light':
                        if is_head_on:
                            y_min, y_max = int(h * 0.25), int(h * 0.65)
                            if t_part['rel_x'] > w * 0.5: x_max = int(w * 0.35)
                            else: x_min = int(w * 0.65)
                        else:
                            y_min, y_max = int(h * 0.25), int(h * 0.65)
                            if t_part['rel_x'] < w * 0.5: x_max = int(w * 0.35)
                            else: x_min = int(w * 0.65)
                    elif label == 'Wheel':
                        y_min = int(h * 0.4)
                        if t_part['rel_x'] < w * 0.5: x_max = int(w * 0.5)
                        else: x_min = int(w * 0.5)

                    # Safety expansion to prevent OpenCV crash
                    y_max = max(y_max, th + y_min + 2)
                    x_max = max(x_max, tw + x_min + 2)
                    search_zone = canvas[y_min:y_max, x_min:x_max]

                    # SHAPE MATCHING (Lock and Key)
                    hole_mask = cv2.inRange(search_zone, np.array([0, 0, 0]), np.array([0, 0, 0]))
                    
                    # Match the Donor Silhouette (The Key) to the Chassis Hole
                    res = cv2.matchTemplate(hole_mask, s_mask, cv2.TM_CCORR_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                    
                    fit_x, fit_y = x_min + max_loc[0], y_min + max_loc[1]

                    # PASTE THE SWAP
                    if fit_y + th <= canvas.shape[0] and fit_x + tw <= canvas.shape[1]:
                        roi = canvas[fit_y:fit_y+th, fit_x:fit_x+tw]
                        img_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(s_mask))
                        img_fg = cv2.bitwise_and(s_img, s_img, mask=s_mask)
                        canvas[fit_y:fit_y+th, fit_x:fit_x+tw] = cv2.add(img_bg, img_fg)
                        
            # Apply the finished car back to the global image
            self.swapped_cars_img[y:y+h, x:x+w] = canvas


    # ---------------------------------------------------------
    # SIFT FEATURE MATCHING
    # ---------------------------------------------------------
    def pair_and_compare_parts(self, parts_car1, parts_car2, output_dir):
        "Uses Scale-Invariant Feature Transform (SIFT) to visually match corresponding parts."
        print("\n" + "="*50)
        print(" PAIRING AND COMPARING CAR PARTS (SIFT MATCHING)")
        print("="*50)
        
        matches_dir = os.path.join(output_dir, "Matched_Parts")
        os.makedirs(matches_dir, exist_ok=True)
        
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()

        for label in ['Wheel', 'Window', 'Light']:
            c1_items = [p for p in parts_car1 if p['label'] == label]
            c2_items = [p for p in parts_car2 if p['label'] == label]
            
            c1_items.sort(key=lambda p: p['rel_x'])
            c2_items.sort(key=lambda p: p['rel_x'])
            
            for i, (part1, part2) in enumerate(zip(c1_items, c2_items)):
                position = "Left_Rear" if i == 0 else "Right_Front"
                print(f"\n--- Comparing {position.replace('_', '/')} {label}s ---")
                
                img1, img2 = part1['img'], part2['img']
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                
                # Extract keypoints (corners/textures) and descriptors from both parts
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
                
                if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                    print("  -> Not enough texture features to compare.")
                    continue
                
                # Find matching points between the two images
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's Ratio Test (Only keep mathematically strong matches)
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.8 * n.distance:
                            good_matches.append(m)
                
                match_score = len(good_matches)
                print(f"  -> Valid SIFT Matches (Ratio < 0.8): {match_score}")
                
                # Draw visual lines connecting the matching features and save to folder
                match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                match_img_bgr = cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR)
                filepath = os.path.join(matches_dir, f"Match_{label}_{position}.png")
                cv2.imwrite(filepath, match_img_bgr)
                print(f"  -> Saved match image to: {filepath}")

    # ---------------------------------------------------------
    # MASTER EXECUTION LOOP
    # ---------------------------------------------------------
    def extract_all_parts(self, output_dir):
        "The main controller function that triggers extraction, saving, and matching."
        if self.mask_cleaned is None or self.final_enhanced is None:
            raise RuntimeError("Must call create_mask() and enhance_cars() before extracting parts.")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Extracting parts to {output_dir}...")

        # Find all isolated cars in the mask
        contours, _ = cv2.findContours(self.mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = sorted([c for c in contours if cv2.contourArea(c) > 1000], key=cv2.contourArea, reverse=True)[:2]

        if not valid_contours:
            print("No cars detected in the mask.")
            return

        all_extracted_parts = []

        full_h, full_w = self.img.shape[:2]
        self.debug_win_thresh = np.zeros((full_h, full_w), dtype=np.uint8)
        self.debug_tire_thresh = np.zeros((full_h, full_w), dtype=np.uint8)
        self.debug_dog_lights = np.zeros((full_h, full_w), dtype=np.uint8)

        # Process each car individually
        for i, car_cnt in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(car_cnt)
            rect = (x, y, w, h)
            
            car_mask = np.zeros_like(self.mask_cleaned)
            cv2.drawContours(car_mask, [car_cnt], -1, 255, -1)
            
            # Run the segmentation logic
            roi_clean, parts = self._identify_parts(self.img_rgb, car_mask, rect)
            all_extracted_parts.append(parts)

            # Save the segmented pieces to the hard drive
            car_dir = os.path.join(output_dir, f"Car_{i+1}")
            os.makedirs(car_dir, exist_ok=True)
            cv2.imwrite(os.path.join(car_dir, "0_Full_Car.png"), cv2.cvtColor(roi_clean, cv2.COLOR_RGB2BGR))
            
            for j, p in enumerate(parts):
                lbl = p['label'].replace(" ", "_").lower()
                cv2.imwrite(os.path.join(car_dir, f"part_{j}_{lbl}.png"), cv2.cvtColor(p['img'], cv2.COLOR_RGB2BGR))
                
        print(f"--> Part extraction completed successfully for {len(valid_contours)} cars.")

        # GENERATE NEW COMPOSITES
        self.create_swapped_image(all_extracted_parts, valid_contours)
        self.apply_new_background()

        # RUN SIFT MATCHING
        if len(all_extracted_parts) >= 2:
            self.pair_and_compare_parts(all_extracted_parts[0], all_extracted_parts[1], output_dir)

        return all_extracted_parts
    
    # ---------------------------------------------------------
    # PLOTTING & VISUALIZATION
    # ---------------------------------------------------------
    def plot_results(self, all_extracted_parts=None):
        "Plots the full technical story, including the Swapped Parts deliverable."
        fig, axs = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f"Technical Analysis: {self.debug_orientation} Mode", fontsize=16)

        # --- ROW 1: ENHANCEMENT ---
        axs[0, 0].imshow(self.img_rgb); axs[0, 0].set_title("1. Original")
        axs[0, 1].imshow(self.spatial_component); axs[0, 1].set_title("2. Spatial (Lighting)")
        axs[0, 2].imshow(self.frequency_component); axs[0, 2].set_title("3. Frequency (Edges)")
        axs[0, 3].imshow(self.final_enhanced); axs[0, 3].set_title("4. Fused Result")

        # --- ROW 2: SEGMENTATION MATH ---
        mag = 20 * np.log(cv2.magnitude(self.dft_shift[:,:,0], self.dft_shift[:,:,1]) + 1)
        axs[1, 0].imshow(mag, cmap='gray'); axs[1, 0].set_title("5. DFT Spectrum")
        axs[1, 1].imshow(self.mask_cleaned, cmap='gray'); axs[1, 1].set_title("6. Binary Mask")
        edges = cv2.Canny(cv2.cvtColor(self.final_enhanced, cv2.COLOR_RGB2GRAY), 50, 150)
        axs[1, 2].imshow(edges, cmap='gray'); axs[1, 2].set_title("7. Edge Detection Aid")
        dist = cv2.distanceTransform(self.mask_cleaned, cv2.DIST_L2, 5)
        axs[1, 3].imshow(dist, cmap='magma'); axs[1, 3].set_title("8. Distance 'Cores'")

        # --- ROW 3: DETECTION LOGIC & SWAP ---
        axs[2, 0].imshow(self.debug_win_thresh, cmap='gray'); axs[2, 0].set_title("9. Window Logic")
        axs[2, 1].imshow(self.debug_tire_thresh, cmap='gray'); axs[2, 1].set_title("10. Tire Logic")
        axs[2, 2].imshow(self.debug_dog_lights, cmap='gray'); axs[2, 2].set_title("11. Light Logic")
        
        # FILL THE EMPTY SPOT WITH SWAPPED PARTS
        if self.swapped_cars_img is not None:
            axs[2, 3].imshow(self.swapped_cars_img)
            axs[2, 3].set_title("12. Swapped Parts Result")
        else:
            axs[2, 3].axis('off')

        # --- ROW 4: FIDELITY & DELIVERABLES ---
        report = "Metrics Unavailable"
        if all_extracted_parts and len(all_extracted_parts) > 0:
            contours, _ = cv2.findContours(self.mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = sorted([c for c in contours if cv2.contourArea(c) > 1000], key=cv2.contourArea, reverse=True)[:2]
            
            if valid_contours:
                x, y, w, h = cv2.boundingRect(valid_contours[0])
                original_roi = self.img_rgb[y:y+h, x:x+w]
                reconstruction = self.reconstruct_car(all_extracted_parts[0], w, h, template=original_roi)
                
                axs[3, 0].imshow(original_roi); axs[3, 0].set_title("13. Original ROI (Target)")
                axs[3, 1].imshow(reconstruction); axs[3, 1].set_title("14. Reconstructed ROI")
                
                m = ImageQualityMetrics(original_roi, reconstruction)
                report = (f"Fidelity Report:\n"
                          f"PSNR: {m.calculate_psnr():.2f}dB\n"
                          f"SSIM: {m.calculate_ssim():.4f}\n"
                          f"MSE: {m.calculate_mse():.1f}")

        axs[3, 2].text(0.5, 0.5, report, fontsize=11, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        axs[3, 2].set_title("15. Fidelity Metrics")

        if self.new_bg_composite is not None:
            axs[3, 3].imshow(self.new_bg_composite); axs[3, 3].set_title("16. New Background")

        for ax in axs.flatten(): ax.axis('off')
        plt.tight_layout()
        
        grid_path = os.path.join("Extracted_Car_Data", "Final_Technical_Grid.png")
        plt.savefig(grid_path, dpi=300)
        plt.show()

# ==============================================================================
# EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    VIDEO_MODE = False  # Set to True to process a video, False for a single image
    path_to_media = 'cars1.jpg' # Can be an image OR video path
    path_to_bg = 'new_background.jpg' 
    
    if VIDEO_MODE:
        print(f"Scanning Video for cars: {path_to_media}")
        captured_frames = extract_keyframes_from_video(path_to_media)
        print(f"--> Extracted {len(captured_frames)} unique car frames.")
        
        for i, frame in enumerate(captured_frames):
            print(f"\n=====================================")
            print(f" PROCESSING VIDEO KEYFRAME {i+1}/{len(captured_frames)}")
            print(f"=====================================")
            
            # Pass the raw frame array instead of an image_path
            pipeline = FrequencyDomainEnhancer(image_array=frame, bg_image_path=path_to_bg)
            
            pipeline.create_mask(D0_low=15, open_ksize=(9, 9), close_ksize=(36, 36))
            pipeline.enhance_cars(D0_high=30, sharp_strength=0.5)
            
            # Save into dedicated folders and catch the return value
            parts = pipeline.extract_all_parts(f"Extracted_Car_Data_VideoFrame_{i+1}")
            
            # Pass the parts to the results plotter
            pipeline.plot_results(all_extracted_parts=parts)
            
    else:
        print(f"Processing Single Image: {path_to_media}")
        pipeline = FrequencyDomainEnhancer(image_path=path_to_media, bg_image_path=path_to_bg)
        
        pipeline.create_mask(D0_low=15, open_ksize=(9, 9), close_ksize=(36, 36))
        pipeline.enhance_cars(D0_high=30, sharp_strength=0.5)
        parts = pipeline.extract_all_parts("Extracted_Car_Data")
        pipeline.plot_results(all_extracted_parts=parts) # Pass the parts here

