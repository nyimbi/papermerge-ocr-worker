import logging
from enum import Enum
from pathlib import Path
from typing import Any

from ..engines.base import OCREngine, OCREngineType

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
	"""Engine selection strategies."""
	BEST_AVAILABLE = 'best_available'
	FASTEST = 'fastest'
	MOST_ACCURATE = 'most_accurate'
	LANGUAGE_OPTIMIZED = 'language_optimized'
	DOCUMENT_TYPE_OPTIMIZED = 'document_type_optimized'
	COST_OPTIMIZED = 'cost_optimized'


# Language to best engine mapping
LANGUAGE_ENGINE_MAP = {
	'chi_sim': OCREngineType.PADDLEOCR,
	'chi_tra': OCREngineType.PADDLEOCR,
	'jpn': OCREngineType.PADDLEOCR,
	'kor': OCREngineType.PADDLEOCR,
	'ara': OCREngineType.PADDLEOCR,
	'hin': OCREngineType.PADDLEOCR,
	'tha': OCREngineType.PADDLEOCR,
	'vie': OCREngineType.PADDLEOCR,
	'eng': OCREngineType.TESSERACT,
	'deu': OCREngineType.TESSERACT,
	'fra': OCREngineType.TESSERACT,
	'spa': OCREngineType.TESSERACT,
}

# Document type to engine mapping
DOCTYPE_ENGINE_MAP = {
	'handwritten': OCREngineType.QWEN_VL,
	'complex_layout': OCREngineType.QWEN_VL,
	'form': OCREngineType.QWEN_VL,
	'table_heavy': OCREngineType.PADDLEOCR,
	'standard_text': OCREngineType.TESSERACT,
	'high_quality_scan': OCREngineType.TESSERACT,
	'low_quality_scan': OCREngineType.QWEN_VL,
	'mixed_content': OCREngineType.QWEN_VL,
}

# Engine performance characteristics
ENGINE_CHARACTERISTICS = {
	OCREngineType.TESSERACT: {
		'speed': 'fast',
		'accuracy_printed': 'high',
		'accuracy_handwritten': 'low',
		'gpu_required': False,
		'cost': 'free',
		'memory_mb': 500,
	},
	OCREngineType.PADDLEOCR: {
		'speed': 'medium',
		'accuracy_printed': 'very_high',
		'accuracy_handwritten': 'medium',
		'gpu_required': False,
		'cost': 'free',
		'memory_mb': 2000,
	},
	OCREngineType.QWEN_VL: {
		'speed': 'slow',
		'accuracy_printed': 'very_high',
		'accuracy_handwritten': 'high',
		'gpu_required': True,
		'cost': 'compute',
		'memory_mb': 8000,
	},
}


class EngineSelector:
	"""Intelligent OCR engine selection."""

	def __init__(
		self,
		available_engines: dict[OCREngineType, OCREngine] | None = None,
		default_strategy: SelectionStrategy = SelectionStrategy.BEST_AVAILABLE
	):
		self.available_engines = available_engines or {}
		self.default_strategy = default_strategy

	def select_engine(
		self,
		strategy: SelectionStrategy | None = None,
		lang: str = 'eng',
		document_type: str | None = None,
		image_path: Path | None = None,
		quality_hint: str | None = None,
		**kwargs
	) -> OCREngineType:
		"""
		Select the best OCR engine based on strategy and context.

		Args:
			strategy: Selection strategy (defaults to instance default)
			lang: Language code for OCR
			document_type: Hint about document type
			image_path: Path to image (for analysis if needed)
			quality_hint: Hint about scan quality (good, fair, poor)
			**kwargs: Additional context

		Returns:
			Recommended OCREngineType
		"""
		strategy = strategy or self.default_strategy

		if strategy == SelectionStrategy.FASTEST:
			return self._select_fastest()

		elif strategy == SelectionStrategy.MOST_ACCURATE:
			return self._select_most_accurate(lang, document_type, quality_hint)

		elif strategy == SelectionStrategy.LANGUAGE_OPTIMIZED:
			return self._select_for_language(lang)

		elif strategy == SelectionStrategy.DOCUMENT_TYPE_OPTIMIZED:
			return self._select_for_document_type(document_type, quality_hint)

		elif strategy == SelectionStrategy.COST_OPTIMIZED:
			return self._select_cost_optimized()

		else:
			return self._select_best_available()

	def _select_best_available(self) -> OCREngineType:
		"""Select best available engine."""
		priority = [
			OCREngineType.QWEN_VL,
			OCREngineType.PADDLEOCR,
			OCREngineType.TESSERACT
		]

		for engine_type in priority:
			if engine_type in self.available_engines:
				engine = self.available_engines[engine_type]
				if engine.is_available():
					return engine_type

		return OCREngineType.TESSERACT

	def _select_fastest(self) -> OCREngineType:
		"""Select fastest available engine."""
		priority = [
			OCREngineType.TESSERACT,
			OCREngineType.PADDLEOCR,
			OCREngineType.QWEN_VL
		]

		for engine_type in priority:
			if engine_type in self.available_engines:
				if self.available_engines[engine_type].is_available():
					return engine_type

		return OCREngineType.TESSERACT

	def _select_most_accurate(
		self,
		lang: str,
		document_type: str | None,
		quality_hint: str | None
	) -> OCREngineType:
		"""Select most accurate engine for the context."""
		if quality_hint == 'poor' or document_type in ('handwritten', 'mixed_content'):
			if self._is_available(OCREngineType.QWEN_VL):
				return OCREngineType.QWEN_VL

		if lang in ('chi_sim', 'chi_tra', 'jpn', 'kor', 'ara'):
			if self._is_available(OCREngineType.PADDLEOCR):
				return OCREngineType.PADDLEOCR

		if self._is_available(OCREngineType.PADDLEOCR):
			return OCREngineType.PADDLEOCR

		return self._select_best_available()

	def _select_for_language(self, lang: str) -> OCREngineType:
		"""Select best engine for language."""
		recommended = LANGUAGE_ENGINE_MAP.get(lang)

		if recommended and self._is_available(recommended):
			return recommended

		return self._select_best_available()

	def _select_for_document_type(
		self,
		document_type: str | None,
		quality_hint: str | None
	) -> OCREngineType:
		"""Select best engine for document type."""
		if document_type:
			recommended = DOCTYPE_ENGINE_MAP.get(document_type)
			if recommended and self._is_available(recommended):
				return recommended

		if quality_hint == 'poor':
			if self._is_available(OCREngineType.QWEN_VL):
				return OCREngineType.QWEN_VL

		return self._select_best_available()

	def _select_cost_optimized(self) -> OCREngineType:
		"""Select lowest cost engine."""
		priority = [
			OCREngineType.TESSERACT,
			OCREngineType.PADDLEOCR,
			OCREngineType.QWEN_VL
		]

		for engine_type in priority:
			if self._is_available(engine_type):
				return engine_type

		return OCREngineType.TESSERACT

	def _is_available(self, engine_type: OCREngineType) -> bool:
		"""Check if engine is available."""
		if engine_type not in self.available_engines:
			return False
		return self.available_engines[engine_type].is_available()

	def get_engine_recommendation(
		self,
		lang: str = 'eng',
		document_type: str | None = None,
		quality_hint: str | None = None
	) -> dict[str, Any]:
		"""
		Get detailed recommendation with reasoning.

		Returns:
			Dictionary with recommendation and reasoning
		"""
		recommendations = []

		for strategy in SelectionStrategy:
			if strategy == SelectionStrategy.BEST_AVAILABLE:
				continue

			engine_type = self.select_engine(
				strategy=strategy,
				lang=lang,
				document_type=document_type,
				quality_hint=quality_hint
			)

			recommendations.append({
				'strategy': strategy.value,
				'engine': engine_type.value,
				'available': self._is_available(engine_type),
				'characteristics': ENGINE_CHARACTERISTICS.get(engine_type, {})
			})

		best = self.select_engine(
			lang=lang,
			document_type=document_type,
			quality_hint=quality_hint
		)

		return {
			'recommended_engine': best.value,
			'available': self._is_available(best),
			'context': {
				'language': lang,
				'document_type': document_type,
				'quality_hint': quality_hint
			},
			'all_recommendations': recommendations
		}
