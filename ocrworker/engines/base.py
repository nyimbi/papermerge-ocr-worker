from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


class OCREngineType(str, Enum):
	TESSERACT = 'tesseract'
	PADDLEOCR = 'paddleocr'
	QWEN_VL = 'qwen-vl'
	HYBRID = 'hybrid'


@dataclass
class BoundingBox:
	"""Bounding box for text region."""
	x: float
	y: float
	width: float
	height: float

	@property
	def x2(self) -> float:
		return self.x + self.width

	@property
	def y2(self) -> float:
		return self.y + self.height

	def to_dict(self) -> dict:
		return {'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height}


@dataclass
class TextLine:
	"""Single line of recognized text."""
	text: str
	confidence: float
	bbox: BoundingBox | None = None
	words: list['Word'] = field(default_factory=list)


@dataclass
class Word:
	"""Single word of recognized text."""
	text: str
	confidence: float
	bbox: BoundingBox | None = None


@dataclass
class OCRResult:
	"""Result from OCR processing."""
	text: str
	confidence: float
	lines: list[TextLine] = field(default_factory=list)
	language: str = ''
	engine: str = ''
	processing_time_ms: float = 0
	page_number: int = 1
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict:
		return {
			'text': self.text,
			'confidence': self.confidence,
			'lines': [
				{
					'text': line.text,
					'confidence': line.confidence,
					'bbox': line.bbox.to_dict() if line.bbox else None,
					'words': [
						{
							'text': w.text,
							'confidence': w.confidence,
							'bbox': w.bbox.to_dict() if w.bbox else None
						}
						for w in line.words
					]
				}
				for line in self.lines
			],
			'language': self.language,
			'engine': self.engine,
			'processing_time_ms': self.processing_time_ms,
			'page_number': self.page_number,
			'metadata': self.metadata
		}


class OCREngine(ABC):
	"""Abstract base class for OCR engines."""

	engine_type: OCREngineType

	@abstractmethod
	def ocr_image(
		self,
		image_path: Path,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""
		Perform OCR on a single image.

		Args:
			image_path: Path to the image file
			lang: Language code for OCR
			**kwargs: Additional engine-specific options

		Returns:
			OCRResult with extracted text and metadata
		"""
		pass

	@abstractmethod
	def ocr_pdf_page(
		self,
		pdf_path: Path,
		page_number: int,
		output_dir: Path,
		page_uuid: UUID,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""
		Perform OCR on a single PDF page.

		Args:
			pdf_path: Path to the PDF file
			page_number: 1-based page number to process
			output_dir: Directory for output files
			page_uuid: UUID for the output files
			lang: Language code for OCR
			**kwargs: Additional engine-specific options

		Returns:
			OCRResult with extracted text and metadata
		"""
		pass

	@abstractmethod
	def is_available(self) -> bool:
		"""Check if the engine is properly installed and configured."""
		pass

	@abstractmethod
	def supported_languages(self) -> list[str]:
		"""Return list of supported language codes."""
		pass

	def get_engine_info(self) -> dict:
		"""Return engine information."""
		return {
			'type': self.engine_type.value,
			'available': self.is_available(),
			'languages': self.supported_languages() if self.is_available() else []
		}
