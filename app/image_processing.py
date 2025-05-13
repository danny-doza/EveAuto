import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import TesseractError
import os

from rich_log import get_logger
logger = get_logger("ImageProcessor")

class ImageProcessor:

    @staticmethod
    def rootsift(descriptors):
        eps = 1e-7
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        return np.sqrt(descriptors)


    @classmethod
    def _match_sift(cls, template_cv: np.ndarray, screenshot_cv: np.ndarray, scale_factor: float, confidence: float,
                    debug: bool = False):
        sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=3, nOctaveLayers=4)
        kp1, des1 = sift.detectAndCompute(template_cv, None)
        kp2, des2 = sift.detectAndCompute(screenshot_cv, None)

        if des1 is None or des2 is None:
            logger.info(f"Unable to find any keypoint for sift match.")
            return None

        des1 = ImageProcessor.rootsift(des1)
        des2 = ImageProcessor.rootsift(des2)

        if debug:
            img_kp1 = cv2.drawKeypoints(template_cv, kp1, None)
            img_kp2 = cv2.drawKeypoints(screenshot_cv, kp2, None)
            cv2.imwrite("./ocr_debug/template_keypoints.png", img_kp1)
            cv2.imwrite("./ocr_debug/screenshot_keypoints.png", img_kp2)

        if des1 is None or des2 is None:
            logger.warning("No descriptors found.")
            return None

        # Use FLANN-based matcher for better speed and accuracy
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Add brute-force matcher with cross-check for improved accuracy
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        cross_matches = bf_matcher.match(des1, des2)
        cross_matches = sorted(cross_matches, key=lambda x: x.distance)

        # Use top 50 cross-check matches for stability
        good_matches = cross_matches[:50]

        match_ratio = len(good_matches) / min(len(des1), len(des2)) if min(len(des1), len(des2)) > 0 else 0
        logger.debug(f"Cross-check match count: {len(good_matches)}, Match ratio: {match_ratio}")

        if len(good_matches) >= 4 and match_ratio >= confidence:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
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
    def match_template(cls, screenshot_img: Image.Image, template_img: Image.Image, confidence: float = 0.12,
                       scale_factor: float = 3.0,
                       threshold: bool = False, lower_hsv=None, upper_hsv=None,
                       debug: bool = False):
        logger.info("Running SIFT-based template match...")

        # Apply denoising before grayscale
        # screenshot_denoised = cv2.fastNlMeansDenoisingColored(np.array(screenshot_img), None, 10, 10, 7, 21)
        # template_denoised = cv2.fastNlMeansDenoisingColored(np.array(template_img), None, 10, 10, 7, 21)

        # Convert images to grayscale OpenCV format
        screenshot_cv = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_BGR2GRAY)
        screenshot_cv = cv2.resize(screenshot_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        template_cv = cv2.cvtColor(np.array(template_img), cv2.COLOR_BGR2GRAY)
        template_cv = cv2.resize(template_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        if threshold:
            logger.debug("Matching only white pixels")
            # Threshold to isolate white icons
            _, screenshot_thresh = cv2.threshold(screenshot_cv, 100, 255, cv2.THRESH_BINARY)
            # Keep only white regions
            screenshot_thresholded = cv2.bitwise_and(screenshot_cv, screenshot_thresh)

            # Threshold to isolate white icons
            _, template_thresh = cv2.threshold(template_cv, 120, 255, cv2.THRESH_BINARY)
            # Keep only white regions
            template_thresholded = cv2.bitwise_and(template_cv, template_thresh)

        # Connect character parts with a modest elliptical close
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        if threshold:
            screenshot_closed = cv2.morphologyEx(screenshot_thresholded, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            template_closed = cv2.morphologyEx(template_thresholded, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        else:
            screenshot_closed = cv2.morphologyEx(screenshot_cv, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            template_closed = cv2.morphologyEx(template_cv, cv2.MORPH_CLOSE, close_kernel, iterations=1)
       
        screenshot_bilateral = cv2.bilateralFilter(screenshot_closed, 9, 75, 75)
        template_bilateral = cv2.bilateralFilter(template_closed, 9, 75, 75)

        # screenshot_laplacian = cv2.Laplacian(screenshot_bilateral, cv2.CV_64F)
        # screenshot_laplacian = cv2.convertScaleAbs(screenshot_laplacian)
        # template_laplacian = cv2.Laplacian(template_bilateral, cv2.CV_64F)
        # template_laplacian = cv2.convertScaleAbs(template_laplacian)

        # Apply CLAHE
        screenshot_clahe = ImageProcessor.enhance_contrast_grayscale(screenshot_bilateral)
        template_clahe = ImageProcessor.enhance_contrast_grayscale(template_bilateral)

        # Sharpen image
        screenshot_sharpened = ImageProcessor.sharpen(screenshot_clahe)
        template_sharpened = ImageProcessor.sharpen(template_clahe)

        # # Remove small specks and large blobs
        # # Binarize before morphology to protect thin strokes
        # contours, _ = cv2.findContours(screenshot_sharpened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(screenshot_sharpened)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if 5 < area < 5000:  # filter out small noise and large blobs
        #         cv2.drawContours(mask, [cnt], -1, 255, -1)
        # screenshot_result = cv2.bitwise_and(screenshot_sharpened, mask)

        # # Remove small specks and large blobs
        # # Binarize before morphology to protect thin strokes
        _, screenshot_binary = cv2.threshold(screenshot_sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, template_binary = cv2.threshold(template_sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(template_sharpened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(template_sharpened)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if 5 < area < 5000:  # filter out small noise and large blobs
        #         cv2.drawContours(mask, [cnt], -1, 255, -1)
        # template_result = cv2.bitwise_and(template_sharpened, mask)

        if debug:
            os.makedirs("./icon_debug", exist_ok=True)
            os.makedirs("./icon_debug/screenshot", exist_ok=True)
            os.makedirs("./icon_debug/template", exist_ok=True)

            cv2.imwrite("./icon_debug/screenshot/base.png", screenshot_cv)
            cv2.imwrite("./icon_debug/template/base.png", template_cv)

            if threshold:
                cv2.imwrite("./icon_debug/screenshot/thresholded.png", screenshot_thresholded)
                cv2.imwrite("./icon_debug/template/thresholded.png", template_thresholded)

            cv2.imwrite("./icon_debug/screenshot/closed.png", screenshot_closed)
            cv2.imwrite("./icon_debug/template/closed.png", template_closed)


            cv2.imwrite("./icon_debug/screenshot/bilateral.png", screenshot_bilateral)
            cv2.imwrite("./icon_debug/template/bilateral.png", template_bilateral)

            cv2.imwrite("./icon_debug/screenshot/clahe.png", screenshot_clahe)
            cv2.imwrite("./icon_debug/template/clahe.png", template_clahe)

            cv2.imwrite("./icon_debug/screenshot/sharpened.png", screenshot_sharpened)
            cv2.imwrite("./icon_debug/template/sharpened.png", template_sharpened)
            
            cv2.imwrite("./icon_debug/screenshot/binary.png", screenshot_binary)
            cv2.imwrite("./icon_debug/template/binary.png", template_binary)

            # cv2.imwrite("./icon_debug/screenshot/result.png", screenshot_result)
            # cv2.imwrite("./icon_debug/template/result.png", template_result)

        return cls._match_sift(template_binary, screenshot_binary, scale_factor, confidence, debug=debug)


    @classmethod
    def match_template_with_color_mask(cls, screenshot_img: Image.Image, template_img: Image.Image,
                                    scale_factor: float = 3.0, confidence: float = 0.3):
        logger.info("Running SIFT with color-masked region...")

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
        logger.info("Preprocessing image...")

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
    def common_text_preprocess_steps(image: Image.Image, title: str = None,
                                     scale_factor: float = 2.0,
                                     threshold: int = 124,
                                     debug: bool = False) -> np.ndarray:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if scale_factor != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        if threshold:
            logger.debug("Matching only white pixels")
            # Threshold to isolate white icons
            _, image_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            # Keep only white regions
            thresholded = cv2.bitwise_and(gray, image_thresh)

        #bilateral = cv2.bilateralFilter(thresholded, 9, 75, 75)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if threshold:
            clahe = clahe.apply(thresholded)
        else:
            clahe = clahe.apply(gray)
       

        # Remove small specks with a small elliptical open
        # open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # # Connect character parts with a modest elliptical close
        # close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        sharpened = ImageProcessor.sharpen(clahe)

        # Remove small specks and large blobs
        # Binarize before morphology to protect thin strokes
        # _, binary = cv2.threshold(thresholded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(binary)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if 20 < area < 5000:  # filter out small noise and large blobs
        #         cv2.drawContours(mask, [cnt], -1, 255, -1)

        # result = cv2.bitwise_and(gray, mask)

        if debug:
            os.makedirs(f"./ocr_debug/text/{title}", exist_ok=True)
            
            if threshold:
                cv2.imwrite(f"./ocr_debug/text/{title}/thresholded.png", thresholded)
            #cv2.imwrite(f"./ocr_debug/text/{title}/bilateral.png", bilateral)
            cv2.imwrite(f"./ocr_debug/text/{title}/clahe.png", clahe)
            #cv2.imwrite(f"./ocr_debug/text/{title}/closed.png", closed)
            #cv2.imwrite(f"./ocr_debug/text/{title}/opened.png", opened)
            cv2.imwrite(f"./ocr_debug/text/{title}/sharpened.png", sharpened)
            #cv2.imwrite(f"./ocr_debug/text/{title}/result.png", result)

        return sharpened


    @staticmethod
    def preprocess_for_text_ocr(image: Image.Image, title: str = None,
                                scale_factor: float = 2.0,
                                threshold: int = 124, lower_hsv=None, upper_hsv=None,
                                debug: bool = False) -> np.ndarray:
        """
        Preprocess bubble or outlined text for OCR:
        - Rescales
        - Converts to grayscale
        - Enhances contrast
        - Applies closing to bridge outline gaps
        - Removes text halo
        - Uses adaptive thresholding and cleaning
        """

        if lower_hsv is not None and upper_hsv is not None:
            # Apply adaptive thresholding
            accented_regions = ImageProcessor.get_accent_fill_regions(image, lower_hsv, upper_hsv, debug=True)

            sharpened = ImageProcessor.common_text_preprocess_steps(
                accented_regions, title=title,
                scale_factor=scale_factor,
                threshold=threshold,
                debug=True
            )
        else:
            sharpened = ImageProcessor.common_text_preprocess_steps(
                image, title=title,
                scale_factor=scale_factor,
                threshold=threshold,
                debug=True
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(sharpened, kernel, iterations=1)
        inverted = cv2.bitwise_not(dilated)

        if debug:
            os.makedirs(f"./ocr_debug/text/{title}", exist_ok=True)
            
            cv2.imwrite(f"./ocr_debug/text/{title}/dilated.png", dilated)
            cv2.imwrite(f"./ocr_debug/text/{title}/inverted.png", inverted)

        #cleaned = ImageProcessor.remove_text_halo(sharpened)
        #adaptive = ImageProcessor.threshold_and_clean(cleaned, invert=True)

        return inverted


    @staticmethod
    def preprocess_white_text(image: Image.Image, scale_factor: float = 2.0, title: str = None, debug: bool = False) -> np.ndarray:
        """
        Preprocess image to isolate only highlighted (white/light-colored) text.
        Uses a simple brightness threshold on the raw grayscale image.
        """
        # Convert PIL image to OpenCV BGR then to grayscale
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Resize for better detection, if requested
        if scale_factor != 1.0:
            gray = cv2.resize(
                gray,
                (int(gray.shape[1] * scale_factor), int(gray.shape[0] * scale_factor)),
                interpolation=cv2.INTER_CUBIC
            )

        # Threshold bright pixels (white text)
        _, image_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # Keep only white regions
        thresholded_mask = cv2.bitwise_and(gray, image_thresh)

        # Clean up noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if debug:
            os.makedirs("./ocr_debug", exist_ok=True)
            cv2.imwrite(f"./ocr_debug/text_{title}_gray.png", gray)
            cv2.imwrite(f"./ocr_debug/text_{title}_threshold_mask.png", thresholded_mask)
            cv2.imwrite(f"./ocr_debug/text_{title}_morph_mask.png", morph_mask)

        return morph_mask


    def extract_text_data(image: Image, title: str = None,
                          scale_factor: float = 2.0,
                          threshold: int = 124, lower_hsv=None, upper_hsv=None,
                          debug: bool = False):
        
        preprocessed = ImageProcessor.preprocess_for_text_ocr(
            image, title=title,
            scale_factor=scale_factor,
            threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
            debug=debug
        )
        
        if debug:
            cv2.imwrite(f"./ocr_debug/preprocessed_{title}_ocr_img.png", preprocessed)

        try:
            custom_config = (
                '--oem 3 --psm 11 '
            )
            
            data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT, config=custom_config)
            if debug:
                logger.debug(pytesseract.image_to_string(preprocessed, config=custom_config))
        except (TesseractError, UnicodeDecodeError) as e:
            logger.error(f"Tesseract failed: {e}")
            return None

        return data


    @staticmethod
    def get_accent_fill_regions(img: Image.Image, lower_hsv, upper_hsv, min_area: int = 100, debug: bool = False) -> np.ndarray:
        """
        Returns a mask image with only the detected accent-fill regions visible,
        preserving the original image size.
        """
        img_np = np.array(img)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        # Get accent color mask
        blue_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No accent-colored contours found.")
            return np.zeros_like(img_np)  # Return blank image of same size

        # Create blank mask to draw filtered regions
        mask = np.zeros_like(blue_mask)

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill valid region

        # Apply mask to original image
        result = cv2.bitwise_and(img_np, img_np, mask=mask)

        if debug:
            os.makedirs("./inventory_debug/regions", exist_ok=True)
            cv2.imwrite("./inventory_debug/regions/blue_mask.png", blue_mask)
            cv2.imwrite("./inventory_debug/regions/filled_regions.png", result)

        return Image.fromarray(result)


    @staticmethod
    def compute_fill_percentage(img: Image, accent_lower_hsv, accent_upper_hsv, debug: bool = False):
        img = np.array(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        # === STEP 1: Mask accent fill to get inventory bar ===
        blue_mask = cv2.inRange(hsv, accent_lower_hsv, accent_upper_hsv)

        # === STEP 2: Get bounding box of blue fill ===
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No blue contours found.")
            return None

        blue_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(blue_contour)

        # === STEP 3: Extract horizontal bar row ===
        bar_slice = img[y:y+h, x-20:x+w+100]  # slightly wider slice to catch black edges

        # === STEP 4: Convert slice to grayscale and threshold dark regions (black bar) ===
        gray_slice = cv2.cvtColor(bar_slice, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray_slice, 10, 255, cv2.THRESH_BINARY_INV)  # isolate dark pixels

        # === STEP 5: Find black bar width ===
        black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not black_contours:
            logger.warning("No black contours found.")
            return None
        
        black_contour = max(black_contours, key=cv2.contourArea)
        _, _, black_w, _ = cv2.boundingRect(black_contour)
        full_w = w + black_w

        fill_percentage = (w / full_w) * 100
        logger.debug("Fill percentage: {:.2f}%".format(fill_percentage))

        if debug:
            os.makedirs("./inventory_debug", exist_ok=True)

            cv2.imwrite("./inventory_debug/blue_mask.png", blue_mask)
            cv2.imwrite("./inventory_debug/bar_slice.png", bar_slice)
            cv2.imwrite("./inventory_debug/black_mask.png", black_mask)

        return round(fill_percentage, 2)