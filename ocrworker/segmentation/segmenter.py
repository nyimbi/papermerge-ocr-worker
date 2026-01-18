# (c) Copyright Datacraft, 2026
"""Document segmentation - detect and split multiple documents from a single scan."""
import io
import logging
import re
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .models import (
	DocumentBoundary,
	SegmentedDocument,
	SegmentationMethod,
	SegmentationResult,
)
from . import prompts

logger = logging.getLogger(__name__)


class DocumentSegmenter:
	"""Detect and split multiple documents from a single scanned image.

	Supports multiple detection methods:
	- VLM (Vision-Language Model): Most accurate, uses Ollama/Qwen-VL
	- Edge detection: Traditional computer vision approach
	- Contour analysis: OpenCV-based contour detection
	- Hybrid: Combination of methods for best results

	Example usage:
		segmenter = DocumentSegmenter(vlm_client=ollama_client)
		result = await segmenter.segment(image_path)

		if result.multiple_documents_detected:
			for segment in result.segments:
				process_document(segment.image_path)
	"""

	# Minimum confidence threshold for accepting a detected boundary
	MIN_CONFIDENCE_THRESHOLD = 0.6

	# Minimum document area as percentage of total image
	MIN_DOCUMENT_AREA_PERCENT = 5.0

	# Maximum skew angle to auto-correct (degrees)
	MAX_AUTO_DESKEW_ANGLE = 15.0

	def __init__(
		self,
		vlm_client: Any | None = None,
		method: SegmentationMethod = SegmentationMethod.HYBRID,
		output_dir: Path | None = None,
		deskew: bool = True,
		min_confidence: float = 0.6,
	):
		"""Initialize the document segmenter.

		Args:
			vlm_client: OllamaClient instance for VLM-based detection
			method: Primary detection method to use
			output_dir: Directory to save extracted segment images
			deskew: Whether to auto-deskew extracted segments
			min_confidence: Minimum confidence threshold for boundaries
		"""
		self.vlm_client = vlm_client
		self.method = method
		self.output_dir = output_dir or Path('/tmp/segments')
		self.deskew = deskew
		self.min_confidence = min_confidence

		# Lazy-load image processing libraries
		self._pil = None
		self._cv2 = None
		self._np = None

	def _load_pil(self):
		"""Lazy-load PIL."""
		if self._pil is None:
			from PIL import Image
			self._pil = Image
		return self._pil

	def _load_cv2(self):
		"""Lazy-load OpenCV."""
		if self._cv2 is None:
			import cv2
			self._cv2 = cv2
		return self._cv2

	def _load_numpy(self):
		"""Lazy-load numpy."""
		if self._np is None:
			import numpy as np
			self._np = np
		return self._np

	async def segment(
		self,
		image_path: Path | None = None,
		image_bytes: bytes | None = None,
		original_scan_id: UUID | None = None,
	) -> SegmentationResult:
		"""Detect and split multiple documents from a scanned image.

		Args:
			image_path: Path to the image file
			image_bytes: Raw image bytes (alternative to image_path)
			original_scan_id: UUID of the original scan for tracking

		Returns:
			SegmentationResult with extracted document segments
		"""
		start_time = time.time()

		if not image_path and not image_bytes:
			raise ValueError("Either image_path or image_bytes must be provided")

		# Load image
		Image = self._load_pil()
		if image_path:
			image = Image.open(image_path)
		else:
			image = Image.open(io.BytesIO(image_bytes))

		original_width, original_height = image.size

		# Create result object
		result = SegmentationResult(
			original_width=original_width,
			original_height=original_height,
			original_path=image_path,
			method_used=self.method,
		)

		try:
			# Step 1: Detect if multiple documents exist
			multi_doc_info = await self._detect_multiple_documents(image_path, image)

			if not multi_doc_info['is_multiple']:
				# Single document - return as-is with minimal processing
				segment = await self._create_single_segment(
					image,
					image_path,
					original_scan_id,
				)
				result.segments = [segment]
				result.multiple_documents_detected = False

			else:
				# Multiple documents detected
				result.multiple_documents_detected = True
				doc_count = multi_doc_info['count']

				# Step 2: Detect boundaries
				boundaries = await self._detect_boundaries(
					image_path,
					image,
					doc_count,
				)

				if not boundaries:
					# Fallback: Could not detect boundaries, treat as single doc
					result.warnings.append(
						"Could not detect document boundaries, treating as single document"
					)
					segment = await self._create_single_segment(
						image,
						image_path,
						original_scan_id,
					)
					result.segments = [segment]
					result.multiple_documents_detected = False
					result.manual_review_recommended = True

				else:
					# Step 3: Extract and deskew each segment
					result.segments = await self._extract_segments(
						image,
						boundaries,
						original_scan_id,
						image_path,
					)

					# Step 4: Check if manual review is needed
					if any(b.confidence < 0.7 for b in boundaries):
						result.manual_review_recommended = True

		except Exception as e:
			logger.error(f"Segmentation failed: {e}")
			# Fallback to single document
			result.warnings.append(f"Segmentation error: {e}")
			segment = await self._create_single_segment(
				image,
				image_path,
				original_scan_id,
			)
			result.segments = [segment]
			result.manual_review_recommended = True

		result.processing_time_ms = (time.time() - start_time) * 1000
		return result

	async def _detect_multiple_documents(
		self,
		image_path: Path | None,
		image: Any,
	) -> dict[str, Any]:
		"""Detect if image contains multiple documents.

		Returns dict with keys: is_multiple, count, confidence, reason
		"""
		if self.method in (SegmentationMethod.VLM, SegmentationMethod.HYBRID):
			if self.vlm_client and image_path:
				return await self._vlm_detect_multiple(image_path)

		if self.method in (SegmentationMethod.EDGE_DETECTION, SegmentationMethod.HYBRID):
			return await self._cv_detect_multiple(image)

		# Fallback: assume single document
		return {'is_multiple': False, 'count': 1, 'confidence': 0.5, 'reason': 'default'}

	async def _vlm_detect_multiple(self, image_path: Path) -> dict[str, Any]:
		"""Use VLM to detect multiple documents."""
		try:
			prompt = prompts.get_multi_document_detection_prompt()
			result = self.vlm_client.generate_with_image(
				prompt=prompt,
				image_path=image_path,
				temperature=0.0,
			)

			# Parse VLM response
			response_text = result.text.strip()
			return self._parse_multi_doc_response(response_text)

		except Exception as e:
			logger.warning(f"VLM multi-doc detection failed: {e}")
			return {'is_multiple': False, 'count': 1, 'confidence': 0.3, 'reason': str(e)}

	async def _cv_detect_multiple(self, image: Any) -> dict[str, Any]:
		"""Use computer vision to detect multiple documents."""
		try:
			cv2 = self._load_cv2()
			np = self._load_numpy()

			# Convert PIL to OpenCV
			cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

			# Apply edge detection
			edges = cv2.Canny(gray, 50, 150, apertureSize=3)

			# Dilate to connect nearby edges
			kernel = np.ones((5, 5), np.uint8)
			dilated = cv2.dilate(edges, kernel, iterations=2)

			# Find contours
			contours, _ = cv2.findContours(
				dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
			)

			# Filter contours by area
			image_area = image.width * image.height
			min_area = image_area * (self.MIN_DOCUMENT_AREA_PERCENT / 100)

			significant_contours = [
				c for c in contours if cv2.contourArea(c) > min_area
			]

			count = len(significant_contours)
			is_multiple = count > 1

			return {
				'is_multiple': is_multiple,
				'count': max(1, count),
				'confidence': 0.7 if is_multiple else 0.8,
				'reason': f"Detected {count} significant contours via edge detection",
			}

		except Exception as e:
			logger.warning(f"CV multi-doc detection failed: {e}")
			return {'is_multiple': False, 'count': 1, 'confidence': 0.3, 'reason': str(e)}

	def _parse_multi_doc_response(self, response: str) -> dict[str, Any]:
		"""Parse VLM response for multi-document detection."""
		result = {
			'is_multiple': False,
			'count': 1,
			'confidence': 0.5,
			'reason': 'Could not parse response',
		}

		lines = response.upper().split('\n')
		for line in lines:
			line = line.strip()
			if line.startswith('MULTIPLE_DOCUMENTS:'):
				value = line.split(':', 1)[1].strip()
				result['is_multiple'] = value == 'YES'
			elif line.startswith('COUNT:'):
				try:
					result['count'] = int(line.split(':', 1)[1].strip())
				except ValueError:
					pass
			elif line.startswith('CONFIDENCE:'):
				try:
					result['confidence'] = float(line.split(':', 1)[1].strip())
				except ValueError:
					pass
			elif line.startswith('REASON:'):
				result['reason'] = line.split(':', 1)[1].strip()

		return result

	async def _detect_boundaries(
		self,
		image_path: Path | None,
		image: Any,
		doc_count: int,
	) -> list[DocumentBoundary]:
		"""Detect boundaries of each document in the image."""
		boundaries = []

		if self.method in (SegmentationMethod.VLM, SegmentationMethod.HYBRID):
			if self.vlm_client and image_path:
				boundaries = await self._vlm_detect_boundaries(image_path, image, doc_count)

		if not boundaries and self.method in (SegmentationMethod.EDGE_DETECTION, SegmentationMethod.CONTOUR, SegmentationMethod.HYBRID):
			boundaries = await self._cv_detect_boundaries(image, doc_count)

		# Validate and filter boundaries
		valid_boundaries = self._validate_boundaries(boundaries, image.width, image.height)

		return valid_boundaries

	async def _vlm_detect_boundaries(
		self,
		image_path: Path,
		image: Any,
		doc_count: int,
	) -> list[DocumentBoundary]:
		"""Use VLM to detect document boundaries."""
		try:
			prompt = prompts.get_boundary_detection_prompt(doc_count)
			result = self.vlm_client.generate_with_image(
				prompt=prompt,
				image_path=image_path,
				temperature=0.0,
			)

			# Parse VLM response into boundaries
			return self._parse_boundary_response(
				result.text,
				image.width,
				image.height,
			)

		except Exception as e:
			logger.warning(f"VLM boundary detection failed: {e}")
			return []

	async def _cv_detect_boundaries(
		self,
		image: Any,
		doc_count: int,
	) -> list[DocumentBoundary]:
		"""Use OpenCV to detect document boundaries."""
		try:
			cv2 = self._load_cv2()
			np = self._load_numpy()

			# Convert PIL to OpenCV
			cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

			# Apply threshold
			_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

			# Morphological operations
			kernel = np.ones((15, 15), np.uint8)
			closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

			# Find contours
			contours, _ = cv2.findContours(
				closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
			)

			# Filter by area and get bounding rectangles
			image_area = image.width * image.height
			min_area = image_area * (self.MIN_DOCUMENT_AREA_PERCENT / 100)

			boundaries = []
			for contour in contours:
				area = cv2.contourArea(contour)
				if area < min_area:
					continue

				# Get bounding rectangle
				x, y, w, h = cv2.boundingRect(contour)

				# Get rotation angle using minAreaRect
				rect = cv2.minAreaRect(contour)
				angle = rect[2]
				if angle < -45:
					angle = 90 + angle

				# Calculate confidence based on rectangularity
				rect_area = w * h
				rectangularity = area / rect_area if rect_area > 0 else 0
				confidence = min(1.0, rectangularity + 0.3)

				boundaries.append(DocumentBoundary(
					x=x,
					y=y,
					width=w,
					height=h,
					confidence=confidence,
					rotation_angle=angle,
					method=SegmentationMethod.CONTOUR,
				))

			# Sort by position (top-to-bottom, left-to-right)
			boundaries.sort(key=lambda b: (b.y, b.x))

			# Limit to expected document count
			return boundaries[:doc_count]

		except Exception as e:
			logger.warning(f"CV boundary detection failed: {e}")
			return []

	def _parse_boundary_response(
		self,
		response: str,
		image_width: int,
		image_height: int,
	) -> list[DocumentBoundary]:
		"""Parse VLM boundary detection response."""
		boundaries = []

		# Split by document marker
		doc_sections = re.split(r'---DOCUMENT \d+---', response, flags=re.IGNORECASE)

		for section in doc_sections:
			section = section.strip()
			if not section:
				continue

			try:
				boundary_data = {
					'x_pct': 0.0,
					'y_pct': 0.0,
					'width_pct': 100.0,
					'height_pct': 100.0,
					'rotation': 0.0,
					'doc_type': None,
					'confidence': 0.7,
				}

				lines = section.split('\n')
				for line in lines:
					line = line.strip().upper()
					if ':' not in line:
						continue

					key, value = line.split(':', 1)
					key = key.strip()
					value = value.strip()

					try:
						if key == 'X':
							boundary_data['x_pct'] = float(value.replace('%', ''))
						elif key == 'Y':
							boundary_data['y_pct'] = float(value.replace('%', ''))
						elif key == 'WIDTH':
							boundary_data['width_pct'] = float(value.replace('%', ''))
						elif key == 'HEIGHT':
							boundary_data['height_pct'] = float(value.replace('%', ''))
						elif key == 'ROTATION':
							boundary_data['rotation'] = float(value.replace('DEGREES', '').strip())
						elif key == 'TYPE':
							boundary_data['doc_type'] = value.lower()
						elif key == 'CONFIDENCE':
							boundary_data['confidence'] = float(value)
					except ValueError:
						continue

				# Convert percentages to pixels
				x = int((boundary_data['x_pct'] / 100) * image_width)
				y = int((boundary_data['y_pct'] / 100) * image_height)
				width = int((boundary_data['width_pct'] / 100) * image_width)
				height = int((boundary_data['height_pct'] / 100) * image_height)

				# Ensure valid dimensions
				width = max(1, min(width, image_width - x))
				height = max(1, min(height, image_height - y))

				boundaries.append(DocumentBoundary(
					x=x,
					y=y,
					width=width,
					height=height,
					confidence=boundary_data['confidence'],
					rotation_angle=boundary_data['rotation'],
					method=SegmentationMethod.VLM,
					document_type_hint=boundary_data['doc_type'],
				))

			except Exception as e:
				logger.warning(f"Failed to parse boundary section: {e}")
				continue

		return boundaries

	def _validate_boundaries(
		self,
		boundaries: list[DocumentBoundary],
		image_width: int,
		image_height: int,
	) -> list[DocumentBoundary]:
		"""Validate and filter detected boundaries."""
		valid = []
		image_area = image_width * image_height
		min_area = image_area * (self.MIN_DOCUMENT_AREA_PERCENT / 100)

		for boundary in boundaries:
			# Check confidence threshold
			if boundary.confidence < self.min_confidence:
				continue

			# Check minimum area
			if boundary.area < min_area:
				continue

			# Check boundary is within image
			if boundary.x < 0 or boundary.y < 0:
				continue
			if boundary.x + boundary.width > image_width:
				continue
			if boundary.y + boundary.height > image_height:
				continue

			# Check for overlap with already-added boundaries
			has_overlap = any(
				boundary.overlaps_with(v, threshold=0.5)
				for v in valid
			)
			if has_overlap:
				continue

			valid.append(boundary)

		return valid

	async def _create_single_segment(
		self,
		image: Any,
		image_path: Path | None,
		original_scan_id: UUID | None,
	) -> SegmentedDocument:
		"""Create a segment for a single-document image."""
		# Optionally detect and correct skew
		deskew_angle = 0.0
		was_deskewed = False

		if self.deskew:
			deskew_angle = await self._detect_skew(image_path, image)
			if abs(deskew_angle) > 0.5 and abs(deskew_angle) < self.MAX_AUTO_DESKEW_ANGLE:
				image = self._deskew_image(image, deskew_angle)
				was_deskewed = True

		# Save processed image
		segment_id = uuid4()
		output_path = self._save_segment(image, segment_id)

		return SegmentedDocument(
			id=segment_id,
			segment_number=1,
			total_segments=1,
			image_path=output_path,
			original_scan_id=original_scan_id,
			original_scan_path=image_path,
			width=image.width,
			height=image.height,
			was_deskewed=was_deskewed,
			deskew_angle=deskew_angle,
		)

	async def _extract_segments(
		self,
		image: Any,
		boundaries: list[DocumentBoundary],
		original_scan_id: UUID | None,
		original_path: Path | None,
	) -> list[SegmentedDocument]:
		"""Extract document segments based on detected boundaries."""
		segments = []
		total = len(boundaries)

		for i, boundary in enumerate(boundaries, start=1):
			try:
				# Crop image to boundary
				cropped = image.crop((
					boundary.x,
					boundary.y,
					boundary.x + boundary.width,
					boundary.y + boundary.height,
				))

				# Deskew if needed
				deskew_angle = 0.0
				was_deskewed = False
				if self.deskew and abs(boundary.rotation_angle) > 0.5:
					if abs(boundary.rotation_angle) < self.MAX_AUTO_DESKEW_ANGLE:
						cropped = self._deskew_image(cropped, boundary.rotation_angle)
						deskew_angle = boundary.rotation_angle
						was_deskewed = True

				# Save segment
				segment_id = uuid4()
				output_path = self._save_segment(cropped, segment_id)

				segment = SegmentedDocument(
					id=segment_id,
					segment_number=i,
					total_segments=total,
					image_path=output_path,
					boundary=boundary,
					original_scan_id=original_scan_id,
					original_scan_path=original_path,
					width=cropped.width,
					height=cropped.height,
					was_deskewed=was_deskewed,
					deskew_angle=deskew_angle,
					document_type_hint=boundary.document_type_hint,
				)
				segments.append(segment)

			except Exception as e:
				logger.error(f"Failed to extract segment {i}: {e}")
				continue

		return segments

	async def _detect_skew(
		self,
		image_path: Path | None,
		image: Any,
	) -> float:
		"""Detect skew angle of document."""
		# Try VLM first
		if self.vlm_client and image_path:
			try:
				prompt = prompts.get_skew_detection_prompt()
				result = self.vlm_client.generate_with_image(
					prompt=prompt,
					image_path=image_path,
					temperature=0.0,
				)

				# Parse skew angle from response
				for line in result.text.upper().split('\n'):
					if line.strip().startswith('SKEW_ANGLE:'):
						value = line.split(':', 1)[1].strip()
						return float(value)

			except Exception as e:
				logger.debug(f"VLM skew detection failed: {e}")

		# Fallback to Hough transform
		try:
			cv2 = self._load_cv2()
			np = self._load_numpy()

			cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 50, 150, apertureSize=3)

			lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
			if lines is not None:
				angles = []
				for rho, theta in lines[:20, 0]:
					angle = (theta * 180 / np.pi) - 90
					if abs(angle) < 45:
						angles.append(angle)

				if angles:
					return float(np.median(angles))

		except Exception as e:
			logger.debug(f"CV skew detection failed: {e}")

		return 0.0

	def _deskew_image(self, image: Any, angle: float) -> Any:
		"""Rotate image to correct skew."""
		Image = self._load_pil()

		# Rotate with expand to avoid cropping
		rotated = image.rotate(
			-angle,  # Counter-rotate to correct skew
			expand=True,
			resample=Image.Resampling.BICUBIC,
			fillcolor='white',
		)

		return rotated

	def _save_segment(self, image: Any, segment_id: UUID) -> Path:
		"""Save segment image to output directory."""
		self.output_dir.mkdir(parents=True, exist_ok=True)

		output_path = self.output_dir / f"{segment_id}.png"
		image.save(output_path, format='PNG', optimize=True)

		return output_path
