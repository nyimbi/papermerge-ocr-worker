from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class OCREngineType(str, Enum):
	"""OCR engine types."""
	TESSERACT = 'tesseract'
	PADDLEOCR = 'paddleocr'
	QWEN_VL = 'qwen-vl'
	HYBRID = 'hybrid'


class SelectionStrategy(str, Enum):
	"""Engine selection strategies."""
	BEST_AVAILABLE = 'best_available'
	FASTEST = 'fastest'
	MOST_ACCURATE = 'most_accurate'
	LANGUAGE_OPTIMIZED = 'language_optimized'


class Settings(BaseSettings):
	papermerge__redis__url: str | None = None
	papermerge__main__logging_cfg: Path | None = None
	papermerge__main__media_root: Path = Path(".")
	papermerge__main__prefix: str = ""
	papermerge__database__url: str = "sqlite:////db/db.sqlite3"
	aws_access_key_id: str | None = None
	aws_secret_access_key: str | None = None
	aws_region_name: str | None = None
	papermerge__s3__bucket_name: str | None = None

	# OCR Engine Configuration
	ocr_engine: OCREngineType = OCREngineType.TESSERACT
	ocr_selection_strategy: SelectionStrategy = SelectionStrategy.BEST_AVAILABLE
	ocr_confidence_threshold: float = 0.7
	ocr_auto_select: bool = True

	# Ollama/VLM Configuration
	ollama_base_url: str = "http://localhost:11434"
	ollama_ocr_model: str = "qwen2.5-vl:7b"
	ollama_timeout: float = 120.0

	# PaddleOCR Configuration
	paddle_use_gpu: bool = False
	paddle_use_angle_cls: bool = True

	# Processing Configuration
	ocr_dpi: int = 300
	ocr_deskew: bool = True
	ocr_preview_width: int = 300


@lru_cache()
def get_settings():
	return Settings()
