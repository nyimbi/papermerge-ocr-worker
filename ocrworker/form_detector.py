# (c) Copyright Datacraft, 2026
"""Form detection and field extraction module."""
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
	"""Bounding box for detected elements."""
	x1: int
	y1: int
	x2: int
	y2: int

	@property
	def width(self) -> int:
		return self.x2 - self.x1

	@property
	def height(self) -> int:
		return self.y2 - self.y1

	@property
	def area(self) -> int:
		return self.width * self.height

	def to_dict(self) -> dict:
		return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class DetectedField:
	"""A detected form field."""
	field_type: str  # text, checkbox, radio, signature, date, number
	label: str | None
	value: str | None
	bbox: BoundingBox
	confidence: float
	page_number: int


@dataclass
class DetectedSignature:
	"""A detected signature region."""
	bbox: BoundingBox
	signature_type: str  # handwritten, digital, stamp
	page_number: int
	confidence: float
	image_data: bytes | None = None


@dataclass
class FormDetectionResult:
	"""Result of form detection."""
	is_form: bool
	form_type: str | None = None
	fields: list[DetectedField] = field(default_factory=list)
	signatures: list[DetectedSignature] = field(default_factory=list)
	confidence: float = 0.0
	page_count: int = 1


class FormDetector:
	"""Detect and extract form fields from document images."""

	# Common form field patterns
	FIELD_PATTERNS = {
		"name": [r"name\s*:", r"full\s*name", r"first\s*name", r"last\s*name"],
		"date": [r"date\s*:", r"date\s*of\s*birth", r"dob\s*:", r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"],
		"email": [r"email\s*:", r"e-mail", r"[\w\.-]+@[\w\.-]+\.\w+"],
		"phone": [r"phone\s*:", r"tel\s*:", r"mobile\s*:", r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"],
		"address": [r"address\s*:", r"street\s*:", r"city\s*:", r"zip\s*:", r"postal\s*code"],
		"signature": [r"signature\s*:", r"sign\s*here", r"signed\s*:", r"authorized\s*signature"],
		"checkbox": [r"\[\s*\]", r"\(\s*\)", r"☐", r"☑", r"yes\s*/\s*no"],
		"amount": [r"amount\s*:", r"total\s*:", r"\$\s*[\d,]+\.?\d*", r"€\s*[\d,]+\.?\d*"],
		"account": [r"account\s*#?:", r"account\s*number", r"policy\s*#?:", r"reference\s*#?:"],
	}

	# Signature detection keywords
	SIGNATURE_KEYWORDS = ["signature", "sign here", "signed", "authorized by", "approved by"]

	def __init__(self):
		self._ocr_cache: dict[int, list[dict]] = {}

	def detect_form(
		self,
		image_paths: list[Path],
		ocr_results: list[dict],
	) -> FormDetectionResult:
		"""Detect if document is a form and extract fields."""
		all_fields = []
		all_signatures = []
		form_indicators = 0
		total_confidence = 0.0

		for page_num, (image_path, ocr_data) in enumerate(zip(image_paths, ocr_results), 1):
			# Detect form elements
			page_fields, page_signatures, page_indicators = self._process_page(
				image_path, ocr_data, page_num
			)
			all_fields.extend(page_fields)
			all_signatures.extend(page_signatures)
			form_indicators += page_indicators

		# Determine if this is a form
		is_form = form_indicators >= 3 or len(all_fields) >= 5

		# Calculate overall confidence
		if all_fields:
			total_confidence = sum(f.confidence for f in all_fields) / len(all_fields)

		# Detect form type
		form_type = self._detect_form_type(all_fields, ocr_results)

		return FormDetectionResult(
			is_form=is_form,
			form_type=form_type,
			fields=all_fields,
			signatures=all_signatures,
			confidence=total_confidence,
			page_count=len(image_paths),
		)

	def _process_page(
		self,
		image_path: Path,
		ocr_data: dict,
		page_number: int,
	) -> tuple[list[DetectedField], list[DetectedSignature], int]:
		"""Process a single page for form detection."""
		fields = []
		signatures = []
		form_indicators = 0

		# Get OCR blocks
		blocks = ocr_data.get("blocks", [])
		full_text = " ".join(b.get("text", "") for b in blocks).lower()

		# Check for form indicators
		if any(kw in full_text for kw in ["please fill", "complete this", "application form"]):
			form_indicators += 1

		# Detect fields from text patterns
		for block in blocks:
			text = block.get("text", "")
			bbox_data = block.get("bbox", {})

			if not bbox_data:
				continue

			bbox = BoundingBox(
				x1=bbox_data.get("x1", 0),
				y1=bbox_data.get("y1", 0),
				x2=bbox_data.get("x2", 0),
				y2=bbox_data.get("y2", 0),
			)

			# Match against field patterns
			for field_type, patterns in self.FIELD_PATTERNS.items():
				for pattern in patterns:
					if re.search(pattern, text.lower()):
						# Found a field indicator
						value = self._extract_field_value(text, blocks, block)
						fields.append(DetectedField(
							field_type=field_type,
							label=text.strip(),
							value=value,
							bbox=bbox,
							confidence=0.8,
							page_number=page_number,
						))
						form_indicators += 1
						break

		# Detect signatures
		page_signatures = self._detect_signatures(image_path, blocks, page_number)
		signatures.extend(page_signatures)

		# Detect checkboxes visually
		checkbox_fields = self._detect_checkboxes(image_path, page_number)
		fields.extend(checkbox_fields)

		return fields, signatures, form_indicators

	def _extract_field_value(
		self,
		label_text: str,
		all_blocks: list[dict],
		current_block: dict,
	) -> str | None:
		"""Extract the value for a field based on its label."""
		# Check if value is after colon in same block
		if ":" in label_text:
			parts = label_text.split(":", 1)
			if len(parts) > 1 and parts[1].strip():
				return parts[1].strip()

		# Look for value in adjacent block (to the right or below)
		current_bbox = current_block.get("bbox", {})
		current_y = current_bbox.get("y1", 0)
		current_x2 = current_bbox.get("x2", 0)

		for block in all_blocks:
			if block == current_block:
				continue

			bbox = block.get("bbox", {})
			block_x1 = bbox.get("x1", 0)
			block_y1 = bbox.get("y1", 0)

			# Check if block is to the right and roughly on same line
			if block_x1 > current_x2 and abs(block_y1 - current_y) < 20:
				return block.get("text", "").strip()

		return None

	def _detect_signatures(
		self,
		image_path: Path,
		blocks: list[dict],
		page_number: int,
	) -> list[DetectedSignature]:
		"""Detect signature regions in a page."""
		signatures = []

		# Find signature indicators in text
		for block in blocks:
			text = block.get("text", "").lower()
			bbox_data = block.get("bbox", {})

			if any(kw in text for kw in self.SIGNATURE_KEYWORDS):
				# Signature region is typically below the indicator
				sig_bbox = BoundingBox(
					x1=bbox_data.get("x1", 0),
					y1=bbox_data.get("y2", 0),  # Start below text
					x2=bbox_data.get("x2", 0) + 150,
					y2=bbox_data.get("y2", 0) + 80,
				)

				signatures.append(DetectedSignature(
					bbox=sig_bbox,
					signature_type="handwritten",
					page_number=page_number,
					confidence=0.75,
				))

		# Visual signature detection using image processing
		try:
			visual_signatures = self._detect_signatures_visually(image_path, page_number)
			signatures.extend(visual_signatures)
		except Exception as e:
			logger.warning(f"Visual signature detection failed: {e}")

		return signatures

	def _detect_signatures_visually(
		self,
		image_path: Path,
		page_number: int,
	) -> list[DetectedSignature]:
		"""Detect signatures using computer vision."""
		signatures = []

		try:
			# Read image
			img = cv2.imread(str(image_path))
			if img is None:
				return signatures

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Apply threshold to get binary image
			_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

			# Find contours
			contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for contour in contours:
				# Get bounding rectangle
				x, y, w, h = cv2.boundingRect(contour)

				# Filter by aspect ratio and size (signatures tend to be wide and short)
				aspect_ratio = w / h if h > 0 else 0
				area = w * h

				# Signature characteristics: wide, not too tall, reasonable size
				if 2.0 < aspect_ratio < 8.0 and 1000 < area < 50000 and 20 < h < 100:
					# Check if it looks like handwriting (irregular edges)
					perimeter = cv2.arcLength(contour, True)
					circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

					# Low circularity indicates irregular shape (like handwriting)
					if circularity < 0.3:
						sig_bbox = BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)

						# Extract signature image
						sig_img = img[y:y+h, x:x+w]
						_, sig_encoded = cv2.imencode(".png", sig_img)

						signatures.append(DetectedSignature(
							bbox=sig_bbox,
							signature_type="handwritten",
							page_number=page_number,
							confidence=0.65,
							image_data=sig_encoded.tobytes(),
						))

		except Exception as e:
			logger.error(f"Error in visual signature detection: {e}")

		return signatures

	def _detect_checkboxes(
		self,
		image_path: Path,
		page_number: int,
	) -> list[DetectedField]:
		"""Detect checkboxes using image processing."""
		fields = []

		try:
			img = cv2.imread(str(image_path))
			if img is None:
				return fields

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 50, 150)

			# Find contours
			contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for contour in contours:
				# Approximate contour
				epsilon = 0.02 * cv2.arcLength(contour, True)
				approx = cv2.approxPolyDP(contour, epsilon, True)

				# Check if it's a square (checkbox)
				if len(approx) == 4:
					x, y, w, h = cv2.boundingRect(contour)
					aspect_ratio = w / h if h > 0 else 0

					# Square-ish and small (typical checkbox size)
					if 0.8 < aspect_ratio < 1.2 and 10 < w < 40 and 10 < h < 40:
						# Check if filled
						roi = gray[y:y+h, x:x+w]
						white_ratio = np.mean(roi) / 255

						is_checked = white_ratio < 0.7  # More dark pixels = checked

						bbox = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
						fields.append(DetectedField(
							field_type="checkbox",
							label=None,
							value="checked" if is_checked else "unchecked",
							bbox=bbox,
							confidence=0.7,
							page_number=page_number,
						))

		except Exception as e:
			logger.error(f"Error in checkbox detection: {e}")

		return fields

	def _detect_form_type(
		self,
		fields: list[DetectedField],
		ocr_results: list[dict],
	) -> str | None:
		"""Detect the type of form based on fields and content."""
		# Combine all text
		full_text = ""
		for ocr in ocr_results:
			for block in ocr.get("blocks", []):
				full_text += " " + block.get("text", "")
		full_text = full_text.lower()

		# Form type detection rules
		if any(kw in full_text for kw in ["insurance", "policy", "claim", "beneficiary"]):
			return "insurance"
		elif any(kw in full_text for kw in ["tax", "income", "deduction", "w-2", "1099"]):
			return "tax"
		elif any(kw in full_text for kw in ["employment", "job application", "resume", "position"]):
			return "employment"
		elif any(kw in full_text for kw in ["loan", "mortgage", "credit", "borrower"]):
			return "financial"
		elif any(kw in full_text for kw in ["medical", "patient", "health", "physician", "diagnosis"]):
			return "medical"
		elif any(kw in full_text for kw in ["contract", "agreement", "terms", "conditions"]):
			return "legal"
		elif any(kw in full_text for kw in ["invoice", "bill", "payment", "amount due"]):
			return "invoice"

		return "general"
