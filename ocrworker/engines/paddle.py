import logging
import time
from pathlib import Path
from uuid import UUID

from .base import OCREngine, OCRResult, OCREngineType, TextLine, BoundingBox, Word

logger = logging.getLogger(__name__)

# Lazy load PaddleOCR to avoid import overhead
_paddle_ocr_instance = None


def get_paddle_ocr(lang: str = 'en', use_gpu: bool = False):
	"""Get or create PaddleOCR instance."""
	global _paddle_ocr_instance
	if _paddle_ocr_instance is None:
		try:
			from paddleocr import PaddleOCR
			_paddle_ocr_instance = PaddleOCR(
				use_angle_cls=True,
				lang=lang,
				use_gpu=use_gpu,
				show_log=False
			)
		except ImportError:
			raise ImportError("PaddleOCR is not installed. Install with: pip install paddleocr paddlepaddle")
	return _paddle_ocr_instance


# Language code mapping from ISO 639-3 to PaddleOCR
LANG_MAP = {
	'eng': 'en',
	'deu': 'german',
	'fra': 'fr',
	'spa': 'es',
	'por': 'pt',
	'ita': 'it',
	'nld': 'nl',
	'pol': 'pl',
	'rus': 'ru',
	'jpn': 'japan',
	'kor': 'korean',
	'chi_sim': 'ch',
	'chi_tra': 'chinese_cht',
	'ara': 'ar',
	'hin': 'hi',
	'tha': 'th',
	'vie': 'vi',
}


class PaddleOCREngine(OCREngine):
	"""PaddleOCR engine for high-accuracy OCR."""

	engine_type = OCREngineType.PADDLEOCR

	def __init__(self, use_gpu: bool = False):
		self.use_gpu = use_gpu
		self._ocr = None

	def _get_ocr(self, lang: str = 'en'):
		"""Lazy load PaddleOCR instance."""
		if self._ocr is None:
			paddle_lang = LANG_MAP.get(lang, lang)
			self._ocr = get_paddle_ocr(lang=paddle_lang, use_gpu=self.use_gpu)
		return self._ocr

	def ocr_image(
		self,
		image_path: Path,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""Perform OCR on a single image using PaddleOCR."""
		start_time = time.time()

		try:
			ocr = self._get_ocr(lang)
			result = ocr.ocr(str(image_path), cls=True)

			lines: list[TextLine] = []
			full_text_parts: list[str] = []
			confidences: list[float] = []

			if result and result[0]:
				for line_data in result[0]:
					bbox_points = line_data[0]
					text_info = line_data[1]

					text = text_info[0]
					conf = float(text_info[1])

					x_coords = [p[0] for p in bbox_points]
					y_coords = [p[1] for p in bbox_points]
					bbox = BoundingBox(
						x=min(x_coords),
						y=min(y_coords),
						width=max(x_coords) - min(x_coords),
						height=max(y_coords) - min(y_coords)
					)

					words = [Word(text=w, confidence=conf, bbox=None) for w in text.split()]

					lines.append(TextLine(
						text=text,
						confidence=conf,
						bbox=bbox,
						words=words
					))
					full_text_parts.append(text)
					confidences.append(conf)

			full_text = '\n'.join(full_text_parts)
			avg_confidence = sum(confidences) / len(confidences) if confidences else 0

			processing_time = (time.time() - start_time) * 1000

			return OCRResult(
				text=full_text,
				confidence=avg_confidence,
				lines=lines,
				language=lang,
				engine='paddleocr',
				processing_time_ms=processing_time,
				metadata={'use_gpu': self.use_gpu}
			)

		except Exception as e:
			logger.error(f"PaddleOCR failed for {image_path}: {e}")
			raise

	def ocr_pdf_page(
		self,
		pdf_path: Path,
		page_number: int,
		output_dir: Path,
		page_uuid: UUID,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""Perform OCR on a single PDF page using PaddleOCR."""
		import tempfile
		from pdf2image import convert_from_path

		start_time = time.time()

		try:
			images = convert_from_path(
				pdf_path,
				first_page=page_number,
				last_page=page_number,
				dpi=300
			)

			if not images:
				raise ValueError(f"Could not extract page {page_number} from {pdf_path}")

			with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
				temp_path = Path(temp.name)
				images[0].save(temp_path, 'PNG')

			try:
				result = self.ocr_image(temp_path, lang=lang, **kwargs)
				result.page_number = page_number
				result.metadata['output_dir'] = str(output_dir)
				result.metadata['page_uuid'] = str(page_uuid)

				text_file = output_dir / f"{page_uuid}.txt"
				text_file.write_text(result.text)

				json_file = output_dir / f"{page_uuid}.json"
				import json
				json_file.write_text(json.dumps(result.to_dict(), indent=2))

			finally:
				temp_path.unlink(missing_ok=True)

			result.processing_time_ms = (time.time() - start_time) * 1000
			return result

		except Exception as e:
			logger.error(f"PaddleOCR PDF processing failed for {pdf_path} page {page_number}: {e}")
			raise

	def is_available(self) -> bool:
		"""Check if PaddleOCR is installed."""
		try:
			from paddleocr import PaddleOCR
			return True
		except ImportError:
			return False

	def supported_languages(self) -> list[str]:
		"""Return list of supported language codes."""
		return list(LANG_MAP.keys()) + [
			'en', 'ch', 'german', 'fr', 'es', 'pt', 'it', 'nl', 'pl', 'ru',
			'japan', 'korean', 'ar', 'hi', 'th', 'vi', 'chinese_cht'
		]
