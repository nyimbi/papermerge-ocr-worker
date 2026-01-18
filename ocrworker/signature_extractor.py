# (c) Copyright Datacraft, 2026
"""Signature extraction and processing module."""
import logging
import io
from pathlib import Path
from dataclasses import dataclass
from uuid import UUID, uuid4

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSignature:
	"""An extracted signature."""
	id: UUID
	page_number: int
	bounding_box: dict
	image_png: bytes
	image_svg: str | None
	signature_type: str  # handwritten, digital, stamp, initials
	confidence: float
	signer_name: str | None = None
	extracted_from_label: str | None = None


class SignatureExtractor:
	"""Extract and process signatures from document images."""

	def __init__(self):
		self.min_signature_area = 500
		self.max_signature_area = 100000
		self.min_aspect_ratio = 1.5
		self.max_aspect_ratio = 10.0

	def extract_signatures(
		self,
		image_path: Path,
		signature_regions: list[dict],
		page_number: int = 1,
	) -> list[ExtractedSignature]:
		"""Extract signatures from specified regions."""
		signatures = []

		try:
			img = cv2.imread(str(image_path))
			if img is None:
				logger.error(f"Could not read image: {image_path}")
				return signatures

			for region in signature_regions:
				bbox = region.get("bbox", region.get("bounding_box", {}))
				x1 = int(bbox.get("x1", 0))
				y1 = int(bbox.get("y1", 0))
				x2 = int(bbox.get("x2", 0))
				y2 = int(bbox.get("y2", 0))

				# Ensure valid coordinates
				if x2 <= x1 or y2 <= y1:
					continue

				# Add padding
				padding = 10
				x1 = max(0, x1 - padding)
				y1 = max(0, y1 - padding)
				x2 = min(img.shape[1], x2 + padding)
				y2 = min(img.shape[0], y2 + padding)

				# Extract region
				sig_roi = img[y1:y2, x1:x2]

				# Process and clean the signature
				cleaned = self._clean_signature(sig_roi)

				if cleaned is None:
					continue

				# Convert to PNG
				_, png_data = cv2.imencode(".png", cleaned)

				# Create SVG representation
				svg_data = self._create_svg_from_contours(cleaned)

				# Determine signature type
				sig_type = self._classify_signature(cleaned)

				signatures.append(ExtractedSignature(
					id=uuid4(),
					page_number=page_number,
					bounding_box={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
					image_png=png_data.tobytes(),
					image_svg=svg_data,
					signature_type=sig_type,
					confidence=region.get("confidence", 0.7),
					extracted_from_label=region.get("indicator_text"),
				))

		except Exception as e:
			logger.error(f"Error extracting signatures: {e}")

		return signatures

	def extract_all_signatures(
		self,
		image_path: Path,
		page_number: int = 1,
	) -> list[ExtractedSignature]:
		"""Auto-detect and extract all signatures from a page."""
		signatures = []

		try:
			img = cv2.imread(str(image_path))
			if img is None:
				return signatures

			# Detect signature regions
			regions = self._detect_signature_regions(img)

			for region in regions:
				# Extract and process
				sig_roi = img[region["y1"]:region["y2"], region["x1"]:region["x2"]]
				cleaned = self._clean_signature(sig_roi)

				if cleaned is None:
					continue

				_, png_data = cv2.imencode(".png", cleaned)
				svg_data = self._create_svg_from_contours(cleaned)
				sig_type = self._classify_signature(cleaned)

				signatures.append(ExtractedSignature(
					id=uuid4(),
					page_number=page_number,
					bounding_box=region,
					image_png=png_data.tobytes(),
					image_svg=svg_data,
					signature_type=sig_type,
					confidence=region.get("confidence", 0.6),
				))

		except Exception as e:
			logger.error(f"Error in auto signature extraction: {e}")

		return signatures

	def _clean_signature(self, roi: np.ndarray) -> np.ndarray | None:
		"""Clean and isolate signature from background."""
		if roi.size == 0:
			return None

		# Convert to grayscale
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

		# Apply adaptive threshold
		binary = cv2.adaptiveThreshold(
			gray, 255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY_INV,
			11, 2
		)

		# Remove noise
		kernel = np.ones((2, 2), np.uint8)
		cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
		cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

		# Find contours and filter
		contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if not contours:
			return None

		# Create mask with only significant contours
		mask = np.zeros_like(cleaned)
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 50:  # Filter small noise
				cv2.drawContours(mask, [contour], -1, 255, -1)

		# Apply mask
		result = cv2.bitwise_and(cleaned, mask)

		# Convert to white background with black signature
		result = 255 - result

		return result

	def _detect_signature_regions(self, img: np.ndarray) -> list[dict]:
		"""Detect potential signature regions in an image."""
		regions = []

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

		# Dilate to connect signature strokes
		kernel = np.ones((3, 3), np.uint8)
		dilated = cv2.dilate(binary, kernel, iterations=2)

		# Find contours
		contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			area = w * h
			aspect_ratio = w / h if h > 0 else 0

			# Filter based on signature characteristics
			if (self.min_signature_area < area < self.max_signature_area and
				self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):

				# Calculate complexity (signature vs text)
				roi = binary[y:y+h, x:x+w]
				pixel_density = np.sum(roi > 0) / area

				# Signatures typically have 5-30% density
				if 0.05 < pixel_density < 0.35:
					regions.append({
						"x1": x,
						"y1": y,
						"x2": x + w,
						"y2": y + h,
						"confidence": 0.6,
					})

		return regions

	def _classify_signature(self, sig_img: np.ndarray) -> str:
		"""Classify the type of signature."""
		if sig_img is None or sig_img.size == 0:
			return "unknown"

		# Analyze signature characteristics
		contours, _ = cv2.findContours(
			255 - sig_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)

		if not contours:
			return "unknown"

		# Calculate metrics
		total_area = sum(cv2.contourArea(c) for c in contours)
		total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

		if total_area == 0:
			return "unknown"

		# Complexity ratio - handwritten signatures are more complex
		complexity = total_perimeter ** 2 / (4 * np.pi * total_area) if total_area > 0 else 0

		# Number of strokes
		num_strokes = len(contours)

		# Classify based on characteristics
		if num_strokes <= 2 and complexity < 5:
			return "initials"
		elif complexity > 20:
			return "handwritten"
		elif 5 < complexity < 20:
			return "handwritten"
		else:
			return "stamp"

	def _create_svg_from_contours(self, sig_img: np.ndarray) -> str:
		"""Create SVG path from signature contours."""
		if sig_img is None:
			return ""

		height, width = sig_img.shape[:2]

		# Find contours
		contours, _ = cv2.findContours(
			255 - sig_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)

		if not contours:
			return ""

		# Build SVG paths
		paths = []
		for contour in contours:
			if len(contour) < 3:
				continue

			# Simplify contour
			epsilon = 0.5
			simplified = cv2.approxPolyDP(contour, epsilon, True)

			if len(simplified) < 2:
				continue

			# Create path
			points = simplified.reshape(-1, 2)
			path_d = f"M {points[0][0]} {points[0][1]}"

			for point in points[1:]:
				path_d += f" L {point[0]} {point[1]}"

			path_d += " Z"
			paths.append(f'<path d="{path_d}" fill="black" />')

		svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <g>
    {" ".join(paths)}
  </g>
</svg>'''

		return svg

	def compare_signatures(
		self,
		sig1_png: bytes,
		sig2_png: bytes,
	) -> float:
		"""Compare two signatures and return similarity score (0-1)."""
		try:
			# Decode images
			arr1 = np.frombuffer(sig1_png, np.uint8)
			arr2 = np.frombuffer(sig2_png, np.uint8)

			img1 = cv2.imdecode(arr1, cv2.IMREAD_GRAYSCALE)
			img2 = cv2.imdecode(arr2, cv2.IMREAD_GRAYSCALE)

			if img1 is None or img2 is None:
				return 0.0

			# Resize to same dimensions
			size = (100, 50)
			img1 = cv2.resize(img1, size)
			img2 = cv2.resize(img2, size)

			# Normalize
			img1 = img1.astype(np.float32) / 255.0
			img2 = img2.astype(np.float32) / 255.0

			# Calculate structural similarity
			# Using simple correlation coefficient
			mean1 = np.mean(img1)
			mean2 = np.mean(img2)

			numerator = np.sum((img1 - mean1) * (img2 - mean2))
			denominator = np.sqrt(np.sum((img1 - mean1) ** 2) * np.sum((img2 - mean2) ** 2))

			if denominator == 0:
				return 0.0

			correlation = numerator / denominator

			# Convert to 0-1 range
			similarity = (correlation + 1) / 2

			return float(similarity)

		except Exception as e:
			logger.error(f"Error comparing signatures: {e}")
			return 0.0
