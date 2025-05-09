import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import TesseractError
import os

class ImageProcessor:

    @staticmethod
    def rootsift(descriptors):
        eps = 1e-7
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        return np.sqrt(descriptors)


    @classmethod
    def _match_sift(cls, template_cv: np.ndarray, screenshot_cv: np.ndarray, scale_factor: float, confidence: float):
        sift = cv2.SIFT_create(contrastThreshold=0.005, edgeThreshold=5, nOctaveLayers=7)
        kp1, des1 = sift.detectAndCompute(template_cv, None)
        kp2, des2 = sift.detectAndCompute(screenshot_cv, None)

        if des1 is None or des2 is None:
            print(f"Unable to find any keypoint for sift match.")
            return None

        des1 = ImageProcessor.rootsift(des1)
        des2 = ImageProcessor.rootsift(des2)

        debug = True
        if debug:
            img_kp1 = cv2.drawKeypoints(template_cv, kp1, None)
            img_kp2 = cv2.drawKeypoints(screenshot_cv, kp2, None)
            cv2.imwrite("./ocr_debug/template_keypoints.png", img_kp1)
            cv2.imwrite("./ocr_debug/screenshot_keypoints.png", img_kp2)

        if des1 is None or des2 is None:
            print("No descriptors found.")
            return None

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = [m for m, n in matches if m.distance <= 0.9 * n.distance]
        
        match_ratio = len(good_matches) / len(matches) if matches else 0
        print(f"Good match count: {len(good_matches)}, Match ratio: {match_ratio}")

        if len(good_matches) >= 3 and match_ratio >= confidence:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = template_cv.shape
                center = np.array([[[w / 2, h / 2]]], dtype='float32')
                transformed_center = cv2.perspectiveTransform(center, M)
                x, y = transformed_center[0][0]
                x /= scale_factor
                y /= scale_factor
                return int(x), int(y)
        return None


    @classmethod
    def match_template(cls, screenshot_img: Image.Image, template_img: Image.Image, scale_factor: float = 3.0, confidence: float = 0.12):
        print("Running SIFT-based template match...")

        # Apply denoising before grayscale
        # screenshot_denoised = cv2.fastNlMeansDenoisingColored(np.array(screenshot_img), None, 10, 10, 7, 21)
        # template_denoised = cv2.fastNlMeansDenoisingColored(np.array(template_img), None, 10, 10, 7, 21)

        # Convert images to grayscale OpenCV format
        screenshot_cv = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_BGR2GRAY)
        screenshot_cv = cv2.resize(screenshot_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        template_cv = cv2.cvtColor(np.array(template_img), cv2.COLOR_BGR2GRAY)
        template_cv = cv2.resize(template_cv, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        screenshot_cv = cv2.bilateralFilter(screenshot_cv, 9, 75, 75)
        template_cv = cv2.bilateralFilter(template_cv, 9, 75, 75)

        screenshot_cv = cv2.Laplacian(screenshot_cv, cv2.CV_64F)
        screenshot_cv = cv2.convertScaleAbs(screenshot_cv)
        template_cv = cv2.Laplacian(template_cv, cv2.CV_64F)
        template_cv = cv2.convertScaleAbs(template_cv)

        # # Apply CLAHE
        template_cv = ImageProcessor.enhance_contrast_grayscale(template_cv)
        screenshot_cv = ImageProcessor.enhance_contrast_grayscale(screenshot_cv)

        # # Sharpen image
        template_cv = ImageProcessor.sharpen(template_cv)
        screenshot_cv = ImageProcessor.sharpen(screenshot_cv)

        return cls._match_sift(template_cv, screenshot_cv, scale_factor, confidence)


    @classmethod
    def match_template_with_color_mask(cls, screenshot_img: Image.Image, template_img: Image.Image,
                                    scale_factor: float = 3.0, confidence: float = 0.3):
        print("Running SIFT with color-masked region...")

        # Convert screenshot to HSV
        screenshot_cv = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2HSV)

        # Extract HSV bounds from template
        template_bgr = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2BGR)
        template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(template_hsv)
        mask_nonblack = cv2.inRange(template_bgr, (1, 1, 1), (255, 255, 255))

        h_vals = h[mask_nonblack > 0]
        s_vals = s[mask_nonblack > 0]
        v_vals = v[mask_nonblack > 0]

        lower_hsv = np.array([max(0, int(np.min(h_vals)) - 10),
                            max(0, int(np.min(s_vals)) - 40),
                            max(0, int(np.min(v_vals)) - 40)])
        upper_hsv = np.array([min(179, int(np.max(h_vals)) + 10),
                            min(255, int(np.max(s_vals)) + 40),
                            min(255, int(np.max(v_vals)) + 40)])

        # Mask screenshot and prepare grayscale input
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Resize
        masked_gray = cv2.resize(masked_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        template_gray = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2GRAY)
        template_gray = cv2.resize(template_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        return cls._match_sift(template_gray, masked_gray, scale_factor, confidence)

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)


    @staticmethod
    def remove_text_halo(image: np.ndarray) -> np.ndarray:
        """
        Remove thin outline/hue halos around text by morphological opening.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        # Subtract the halos
        halo = cv2.subtract(image, opened)
        return cv2.subtract(image, halo)


    @staticmethod
    def threshold_and_clean(image: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Apply adaptive threshold and morphological closing to clean text regions.
        invert=True yields white-on-black output.
        """
        method = cv2.ADAPTIVE_THRESH_MEAN_C
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        thresh = cv2.adaptiveThreshold(image, 255, method, thresh_type, 15, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)


    @staticmethod
    def enhance_contrast_grayscale(image: np.ndarray) -> np.ndarray:
        # Convert to HSV to isolate brightness
        hsv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)

        hsv = cv2.merge((h, s, v))
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)


    @classmethod
    def preprocess(cls, image: Image.Image, scale_factor: int = 1) -> Image.Image:
        print("Preprocessing image (simplified)...")

        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if scale_factor != 1:
            gray = cv2.resize(gray, (int(gray.shape[1]*scale_factor), int(gray.shape[0]*scale_factor)), interpolation=cv2.INTER_CUBIC)

        # Apply a light Gaussian blur to smooth noise without destroying edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply Otsu's binarization
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save debug images
        os.makedirs("./ocr_debug", exist_ok=True)
        cv2.imwrite("./ocr_debug/gray.png", gray)
        cv2.imwrite("./ocr_debug/blurred.png", blurred)
        cv2.imwrite("./ocr_debug/otsu_thresh.png", otsu_thresh)

        return Image.fromarray(otsu_thresh)


    @staticmethod
    def preprocess_for_color_match(image: Image.Image) -> np.ndarray:
        """
        Preprocess image for HSV-based color detection.
        Converts image to HSV without modifying contrast or applying blur.
        """
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv


    @staticmethod
    def common_text_preprocess_steps(image: Image.Image, scale_factor: float = 2.0) -> np.ndarray:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if scale_factor != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Remove small specks and large blobs
        _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 5000:  # filter out small noise and large blobs
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        result = cv2.bitwise_and(closed, mask)
        return result


    @staticmethod
    def preprocess_for_text_ocr(image: Image.Image, scale_factor: float = 2.0) -> np.ndarray:
        """
        Preprocess bubble or outlined text for OCR:
        - Rescales
        - Converts to grayscale
        - Enhances contrast
        - Applies closing to bridge outline gaps
        - Removes text halo
        - Uses adaptive thresholding and cleaning
        """
        closed = ImageProcessor.common_text_preprocess_steps(image, scale_factor)
        cleaned = ImageProcessor.remove_text_halo(closed)
        adaptive = ImageProcessor.threshold_and_clean(cleaned, invert=True)

        # Debug output
        os.makedirs("./ocr_debug", exist_ok=True)
        cv2.imwrite("./ocr_debug/text_gray.png", closed)
        cv2.imwrite("./ocr_debug/text_closed.png", closed)
        cv2.imwrite("./ocr_debug/text_thresholded.png", adaptive)

        return adaptive


    @staticmethod
    def preprocess_white_text(image: Image.Image, scale_factor: float = 2.0) -> np.ndarray:
        """
        Preprocess image to isolate only highlighted (white/light-colored) text.
        Strongly suppresses darker text via contrast stretching and pixel masking.
        """
        closed = ImageProcessor.common_text_preprocess_steps(image, scale_factor)
        cleaned = ImageProcessor.remove_text_halo(closed)

        # Contrast stretch and mask bright pixels
        gray = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
        adaptive = ImageProcessor.threshold_and_clean(gray, invert=False)
        return adaptive


    def extract_text_data(image: Image, scale_factor: float = 2.0, highlighted: bool = False):
        if highlighted:
            preprocessed = ImageProcessor.preprocess_white_text(image)
        else:
            preprocessed = ImageProcessor.preprocess_for_text_ocr(image, scale_factor=scale_factor)
        
        debug = True
        if debug:
            cv2.imwrite("./ocr_debug/preprocessed_ocr_img.png", preprocessed)


        try:
            custom_config = (
                '--oem 3 --psm 11 '
            )
            
            data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT, config=custom_config)
            print(pytesseract.image_to_string(preprocessed, config=custom_config))
        except (TesseractError, UnicodeDecodeError) as e:
            print(f"[ERROR] Tesseract failed: {e}")
            return None

        return data