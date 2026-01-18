import logging
import time
from pathlib import Path
from uuid import UUID

from ..engines.base import OCREngine, OCRResult, OCREngineType, TextLine
from ..engines.tesseract import TesseractEngine
from .selector import EngineSelector, SelectionStrategy

logger = logging.getLogger(__name__)


class HybridOCRPipeline(OCREngine):
	"""
	Hybrid OCR pipeline combining multiple engines.

	Features:
	- Intelligent engine selection based on document characteristics
	- Fallback to secondary engines on failure
	- Confidence-based result merging
	- Quality-aware processing
	"""

	engine_type = OCREngineType.HYBRID

	def __init__(
		self,
		primary_engine: OCREngineType = OCREngineType.TESSERACT,
		fallback_engines: list[OCREngineType] | None = None,
		selection_strategy: SelectionStrategy = SelectionStrategy.BEST_AVAILABLE,
		confidence_threshold: float = 0.7,
		auto_select: bool = True,
		ollama_base_url: str = 'http://localhost:11434',
		ollama_model: str = 'qwen2.5-vl:7b',
		use_gpu: bool = False
	):
		self.primary_engine_type = primary_engine
		self.fallback_engines = fallback_engines or [
			OCREngineType.PADDLEOCR,
			OCREngineType.QWEN_VL
		]
		self.selection_strategy = selection_strategy
		self.confidence_threshold = confidence_threshold
		self.auto_select = auto_select
		self.ollama_base_url = ollama_base_url
		self.ollama_model = ollama_model
		self.use_gpu = use_gpu

		self._engines: dict[OCREngineType, OCREngine] = {}
		self._selector: EngineSelector | None = None

	def _get_engine(self, engine_type: OCREngineType) -> OCREngine:
		"""Get or create engine instance."""
		if engine_type not in self._engines:
			from ..engines.factory import get_ocr_engine
			self._engines[engine_type] = get_ocr_engine(
				engine_type,
				ollama_base_url=self.ollama_base_url,
				ollama_model=self.ollama_model,
				use_gpu=self.use_gpu
			)
		return self._engines[engine_type]

	@property
	def selector(self) -> EngineSelector:
		"""Get or create engine selector."""
		if self._selector is None:
			available = {}
			for engine_type in [self.primary_engine_type] + self.fallback_engines:
				try:
					available[engine_type] = self._get_engine(engine_type)
				except Exception as e:
					logger.warning(f"Could not initialize {engine_type}: {e}")

			self._selector = EngineSelector(
				available_engines=available,
				default_strategy=self.selection_strategy
			)
		return self._selector

	def _select_engine(
		self,
		lang: str,
		image_path: Path | None = None,
		**kwargs
	) -> OCREngineType:
		"""Select best engine for the task."""
		if not self.auto_select:
			return self.primary_engine_type

		document_type = kwargs.get('document_type')
		quality_hint = kwargs.get('quality_hint')

		return self.selector.select_engine(
			strategy=self.selection_strategy,
			lang=lang,
			document_type=document_type,
			image_path=image_path,
			quality_hint=quality_hint
		)

	def ocr_image(
		self,
		image_path: Path,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""
		Perform OCR on image using hybrid pipeline.

		The pipeline:
		1. Selects best engine based on context
		2. Runs primary engine
		3. Falls back to secondary engines if confidence is low
		4. Optionally merges results from multiple engines
		"""
		start_time = time.time()

		selected_engine = self._select_engine(lang, image_path, **kwargs)
		engines_tried = []
		best_result: OCRResult | None = None

		all_engines = [selected_engine] + [
			e for e in self.fallback_engines if e != selected_engine
		]

		for engine_type in all_engines:
			try:
				engine = self._get_engine(engine_type)
				if not engine.is_available():
					continue

				result = engine.ocr_image(image_path, lang=lang, **kwargs)
				engines_tried.append(engine_type.value)

				if best_result is None or result.confidence > best_result.confidence:
					best_result = result

				if result.confidence >= self.confidence_threshold:
					break

			except Exception as e:
				logger.warning(f"Engine {engine_type} failed: {e}")
				continue

		if best_result is None:
			logger.error(f"All OCR engines failed for {image_path}")
			return OCRResult(
				text='',
				confidence=0,
				language=lang,
				engine='hybrid',
				processing_time_ms=(time.time() - start_time) * 1000,
				metadata={'error': 'All engines failed', 'engines_tried': engines_tried}
			)

		best_result.metadata['engines_tried'] = engines_tried
		best_result.metadata['selected_engine'] = selected_engine.value
		best_result.metadata['hybrid_pipeline'] = True
		best_result.processing_time_ms = (time.time() - start_time) * 1000

		return best_result

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
		Perform OCR on PDF page using hybrid pipeline.
		"""
		start_time = time.time()

		selected_engine = self._select_engine(lang, **kwargs)
		engines_tried = []
		best_result: OCRResult | None = None

		all_engines = [selected_engine] + [
			e for e in self.fallback_engines if e != selected_engine
		]

		for engine_type in all_engines:
			try:
				engine = self._get_engine(engine_type)
				if not engine.is_available():
					continue

				result = engine.ocr_pdf_page(
					pdf_path, page_number, output_dir, page_uuid, lang=lang, **kwargs
				)
				engines_tried.append(engine_type.value)

				if best_result is None or result.confidence > best_result.confidence:
					best_result = result

				if result.confidence >= self.confidence_threshold:
					break

			except Exception as e:
				logger.warning(f"Engine {engine_type} failed on page {page_number}: {e}")
				continue

		if best_result is None:
			logger.error(f"All OCR engines failed for {pdf_path} page {page_number}")
			return OCRResult(
				text='',
				confidence=0,
				language=lang,
				engine='hybrid',
				page_number=page_number,
				processing_time_ms=(time.time() - start_time) * 1000,
				metadata={'error': 'All engines failed', 'engines_tried': engines_tried}
			)

		best_result.metadata['engines_tried'] = engines_tried
		best_result.metadata['selected_engine'] = selected_engine.value
		best_result.metadata['hybrid_pipeline'] = True
		best_result.processing_time_ms = (time.time() - start_time) * 1000

		return best_result

	def ocr_with_classification(
		self,
		image_path: Path,
		lang: str = 'eng',
		**kwargs
	) -> tuple[OCRResult, dict]:
		"""
		Perform OCR with document classification.

		Returns:
			Tuple of (OCRResult, classification_dict)
		"""
		from ..classification.detector import DocumentClassifier

		try:
			qwen = self._get_engine(OCREngineType.QWEN_VL)
			if qwen.is_available():
				classifier = DocumentClassifier(
					base_url=self.ollama_base_url,
					model=self.ollama_model
				)
				classification = classifier.classify(image_path)
				classification_dict = classification.to_dict()

				document_type = classification.document_type.value
				kwargs['document_type'] = document_type
		except Exception as e:
			logger.warning(f"Classification failed, proceeding without: {e}")
			classification_dict = {'error': str(e)}

		ocr_result = self.ocr_image(image_path, lang=lang, **kwargs)

		return ocr_result, classification_dict

	def is_available(self) -> bool:
		"""Check if at least one engine is available."""
		for engine_type in [self.primary_engine_type] + self.fallback_engines:
			try:
				engine = self._get_engine(engine_type)
				if engine.is_available():
					return True
			except Exception:
				continue
		return False

	def supported_languages(self) -> list[str]:
		"""Return union of languages from all available engines."""
		languages = set()
		for engine_type in [self.primary_engine_type] + self.fallback_engines:
			try:
				engine = self._get_engine(engine_type)
				if engine.is_available():
					languages.update(engine.supported_languages())
			except Exception:
				continue
		return sorted(languages)

	def get_pipeline_info(self) -> dict:
		"""Get detailed pipeline configuration and status."""
		engine_status = {}
		for engine_type in [self.primary_engine_type] + self.fallback_engines:
			try:
				engine = self._get_engine(engine_type)
				engine_status[engine_type.value] = {
					'available': engine.is_available(),
					'languages': engine.supported_languages() if engine.is_available() else []
				}
			except Exception as e:
				engine_status[engine_type.value] = {
					'available': False,
					'error': str(e)
				}

		return {
			'type': 'hybrid',
			'primary_engine': self.primary_engine_type.value,
			'fallback_engines': [e.value for e in self.fallback_engines],
			'selection_strategy': self.selection_strategy.value,
			'confidence_threshold': self.confidence_threshold,
			'auto_select': self.auto_select,
			'engines': engine_status
		}
