# (c) Copyright Datacraft, 2026
"""Data models for document segmentation."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class SegmentationMethod(str, Enum):
	"""Methods for detecting document boundaries."""
	VLM = 'vlm'  # Vision-Language Model analysis
	EDGE_DETECTION = 'edge_detection'  # Traditional CV edge detection
	CONTOUR = 'contour'  # Contour analysis
	HYBRID = 'hybrid'  # Combination of methods
	TEMPLATE = 'template'  # Template matching for known layouts


@dataclass
class DocumentBoundary:
	"""Detected boundary of a document within an image.

	Coordinates are in pixels relative to the original image.
	"""
	x: int  # Left edge
	y: int  # Top edge
	width: int
	height: int
	confidence: float  # 0.0 to 1.0

	# Rotation/skew angle in degrees (positive = clockwise)
	rotation_angle: float = 0.0

	# Detection metadata
	method: SegmentationMethod = SegmentationMethod.HYBRID
	document_type_hint: str | None = None  # e.g., "invoice", "letter"

	def __post_init__(self):
		"""Validate boundary coordinates."""
		if self.width <= 0 or self.height <= 0:
			raise ValueError("Boundary width and height must be positive")
		if not 0.0 <= self.confidence <= 1.0:
			raise ValueError("Confidence must be between 0.0 and 1.0")

	@property
	def area(self) -> int:
		"""Calculate boundary area in pixels."""
		return self.width * self.height

	@property
	def center(self) -> tuple[int, int]:
		"""Calculate center point of boundary."""
		return (self.x + self.width // 2, self.y + self.height // 2)

	@property
	def corners(self) -> list[tuple[int, int]]:
		"""Get corner coordinates as [(x, y), ...]."""
		return [
			(self.x, self.y),  # top-left
			(self.x + self.width, self.y),  # top-right
			(self.x + self.width, self.y + self.height),  # bottom-right
			(self.x, self.y + self.height),  # bottom-left
		]

	def overlaps_with(self, other: 'DocumentBoundary', threshold: float = 0.1) -> bool:
		"""Check if this boundary overlaps with another by more than threshold."""
		# Calculate intersection
		x_left = max(self.x, other.x)
		y_top = max(self.y, other.y)
		x_right = min(self.x + self.width, other.x + other.width)
		y_bottom = min(self.y + self.height, other.y + other.height)

		if x_right < x_left or y_bottom < y_top:
			return False  # No intersection

		intersection_area = (x_right - x_left) * (y_bottom - y_top)
		smaller_area = min(self.area, other.area)

		return (intersection_area / smaller_area) > threshold

	def to_dict(self) -> dict[str, Any]:
		"""Convert to dictionary for serialization."""
		return {
			'x': self.x,
			'y': self.y,
			'width': self.width,
			'height': self.height,
			'confidence': self.confidence,
			'rotation_angle': self.rotation_angle,
			'method': self.method.value,
			'document_type_hint': self.document_type_hint,
		}

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> 'DocumentBoundary':
		"""Create from dictionary."""
		return cls(
			x=data['x'],
			y=data['y'],
			width=data['width'],
			height=data['height'],
			confidence=data['confidence'],
			rotation_angle=data.get('rotation_angle', 0.0),
			method=SegmentationMethod(data.get('method', 'hybrid')),
			document_type_hint=data.get('document_type_hint'),
		)


@dataclass
class SegmentedDocument:
	"""A single document segment extracted from a multi-document scan."""

	id: UUID = field(default_factory=uuid4)
	segment_number: int = 1
	total_segments: int = 1

	# Image data (bytes or path)
	image_data: bytes | None = None
	image_path: Path | None = None

	# Boundary information from original scan
	boundary: DocumentBoundary | None = None

	# Original scan reference
	original_scan_id: UUID | None = None
	original_scan_path: Path | None = None

	# Image dimensions after extraction/deskewing
	width: int = 0
	height: int = 0

	# Processing metadata
	was_deskewed: bool = False
	deskew_angle: float = 0.0

	# Classification hint from segmentation
	document_type_hint: str | None = None

	def __post_init__(self):
		"""Validate segment data."""
		if self.segment_number < 1:
			raise ValueError("segment_number must be >= 1")
		if self.total_segments < 1:
			raise ValueError("total_segments must be >= 1")
		if self.segment_number > self.total_segments:
			raise ValueError("segment_number cannot exceed total_segments")

	@property
	def is_single_document(self) -> bool:
		"""Check if this is from a single-document scan."""
		return self.total_segments == 1

	@property
	def confidence(self) -> float:
		"""Get segmentation confidence (from boundary or 1.0 if single doc)."""
		if self.boundary:
			return self.boundary.confidence
		return 1.0

	def to_dict(self) -> dict[str, Any]:
		"""Convert to dictionary for serialization."""
		return {
			'id': str(self.id),
			'segment_number': self.segment_number,
			'total_segments': self.total_segments,
			'image_path': str(self.image_path) if self.image_path else None,
			'boundary': self.boundary.to_dict() if self.boundary else None,
			'original_scan_id': str(self.original_scan_id) if self.original_scan_id else None,
			'original_scan_path': str(self.original_scan_path) if self.original_scan_path else None,
			'width': self.width,
			'height': self.height,
			'was_deskewed': self.was_deskewed,
			'deskew_angle': self.deskew_angle,
			'document_type_hint': self.document_type_hint,
		}


@dataclass
class SegmentationResult:
	"""Result of document segmentation operation."""

	# List of detected/extracted segments
	segments: list[SegmentedDocument] = field(default_factory=list)

	# Original image info
	original_width: int = 0
	original_height: int = 0
	original_path: Path | None = None

	# Detection metadata
	method_used: SegmentationMethod = SegmentationMethod.HYBRID
	processing_time_ms: float = 0.0

	# Flags
	multiple_documents_detected: bool = False
	manual_review_recommended: bool = False

	# Any warnings or issues
	warnings: list[str] = field(default_factory=list)

	# Raw detection data (for debugging/logging)
	raw_vlm_response: str | None = None
	raw_detection_data: dict[str, Any] | None = None

	@property
	def document_count(self) -> int:
		"""Number of documents detected."""
		return len(self.segments)

	@property
	def avg_confidence(self) -> float:
		"""Average confidence across all segments."""
		if not self.segments:
			return 0.0
		return sum(s.confidence for s in self.segments) / len(self.segments)

	@property
	def needs_review(self) -> bool:
		"""Check if any segment has low confidence or review is recommended."""
		if self.manual_review_recommended:
			return True
		return any(s.confidence < 0.7 for s in self.segments)

	def to_dict(self) -> dict[str, Any]:
		"""Convert to dictionary for serialization."""
		return {
			'segments': [s.to_dict() for s in self.segments],
			'original_width': self.original_width,
			'original_height': self.original_height,
			'original_path': str(self.original_path) if self.original_path else None,
			'method_used': self.method_used.value,
			'processing_time_ms': self.processing_time_ms,
			'multiple_documents_detected': self.multiple_documents_detected,
			'manual_review_recommended': self.manual_review_recommended,
			'document_count': self.document_count,
			'avg_confidence': self.avg_confidence,
			'warnings': self.warnings,
		}
