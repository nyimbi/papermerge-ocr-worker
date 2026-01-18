import logging
from functools import lru_cache

from .base import OCREngine, OCREngineType

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_ocr_engine(
	engine_type: OCREngineType | str = OCREngineType.TESSERACT,
	**kwargs
) -> OCREngine:
	"""
	Factory function to get an OCR engine instance.

	Args:
		engine_type: Type of OCR engine to create
		**kwargs: Engine-specific configuration options

	Returns:
		OCREngine instance

	Raises:
		ValueError: If engine type is not supported
	"""
	if isinstance(engine_type, str):
		try:
			engine_type = OCREngineType(engine_type)
		except ValueError:
			raise ValueError(f"Unknown engine type: {engine_type}")

	if engine_type == OCREngineType.TESSERACT:
		from .tesseract import TesseractEngine
		return TesseractEngine()

	elif engine_type == OCREngineType.PADDLEOCR:
		from .paddle import PaddleOCREngine
		use_gpu = kwargs.get('use_gpu', False)
		return PaddleOCREngine(use_gpu=use_gpu)

	elif engine_type == OCREngineType.QWEN_VL:
		from .qwen_vl import QwenVLEngine
		return QwenVLEngine(
			base_url=kwargs.get('ollama_base_url', 'http://localhost:11434'),
			model=kwargs.get('ollama_model', 'qwen2.5-vl:7b'),
			timeout=kwargs.get('timeout', 120.0)
		)

	elif engine_type == OCREngineType.HYBRID:
		from ..pipeline.hybrid import HybridOCRPipeline
		return HybridOCRPipeline(**kwargs)

	else:
		raise ValueError(f"Unsupported engine type: {engine_type}")


def get_available_engines() -> dict[str, dict]:
	"""
	Get information about all available OCR engines.

	Returns:
		Dictionary mapping engine type to availability info
	"""
	engines = {}

	for engine_type in OCREngineType:
		if engine_type == OCREngineType.HYBRID:
			engines[engine_type.value] = {
				'type': engine_type.value,
				'available': True,
				'languages': [],
				'description': 'Hybrid pipeline combining multiple engines'
			}
			continue

		try:
			engine = get_ocr_engine(engine_type)
			engines[engine_type.value] = {
				'type': engine_type.value,
				'available': engine.is_available(),
				'languages': engine.supported_languages() if engine.is_available() else [],
				'description': engine.__class__.__doc__ or ''
			}
		except Exception as e:
			logger.warning(f"Could not check engine {engine_type}: {e}")
			engines[engine_type.value] = {
				'type': engine_type.value,
				'available': False,
				'languages': [],
				'error': str(e)
			}

	return engines


def get_best_available_engine(**kwargs) -> OCREngine:
	"""
	Get the best available OCR engine based on availability and capabilities.

	Priority order:
	1. Qwen-VL (if Ollama available)
	2. PaddleOCR (if installed)
	3. Tesseract (fallback)

	Returns:
		Best available OCREngine instance
	"""
	try:
		qwen = get_ocr_engine(OCREngineType.QWEN_VL, **kwargs)
		if qwen.is_available():
			logger.info("Using Qwen-VL OCR engine")
			return qwen
	except Exception:
		pass

	try:
		paddle = get_ocr_engine(OCREngineType.PADDLEOCR, **kwargs)
		if paddle.is_available():
			logger.info("Using PaddleOCR engine")
			return paddle
	except Exception:
		pass

	logger.info("Using Tesseract OCR engine (fallback)")
	return get_ocr_engine(OCREngineType.TESSERACT)
