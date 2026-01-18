import logging
import time
import tempfile
from pathlib import Path
from uuid import UUID

from .base import OCREngine, OCRResult, OCREngineType, TextLine
from ..vlm.client import OllamaClient

logger = logging.getLogger(__name__)


class QwenVLEngine(OCREngine):
	"""Qwen2-VL OCR engine using Ollama."""

	engine_type = OCREngineType.QWEN_VL

	def __init__(
		self,
		base_url: str = 'http://localhost:11434',
		model: str = 'qwen2.5-vl:7b',
		timeout: float = 120.0
	):
		self.base_url = base_url
		self.model = model
		self.timeout = timeout
		self._client: OllamaClient | None = None

	@property
	def client(self) -> OllamaClient:
		"""Lazy-loaded Ollama client."""
		if self._client is None:
			self._client = OllamaClient(
				base_url=self.base_url,
				model=self.model,
				timeout=self.timeout
			)
		return self._client

	def ocr_image(
		self,
		image_path: Path,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""Perform OCR on a single image using Qwen-VL."""
		start_time = time.time()

		try:
			detailed = kwargs.get('detailed', False)
			result = self.client.ocr_image(image_path, lang=lang, detailed=detailed)

			text = result.text.strip()
			lines = [TextLine(text=line, confidence=0.95) for line in text.split('\n') if line.strip()]

			processing_time = (time.time() - start_time) * 1000

			return OCRResult(
				text=text,
				confidence=0.95,
				lines=lines,
				language=lang,
				engine='qwen-vl',
				processing_time_ms=processing_time,
				metadata={
					'model': result.model,
					'prompt_tokens': result.prompt_eval_count,
					'completion_tokens': result.eval_count,
					'vlm_duration_ms': result.eval_duration_ms
				}
			)

		except Exception as e:
			logger.error(f"Qwen-VL OCR failed for {image_path}: {e}")
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
		"""Perform OCR on a single PDF page using Qwen-VL."""
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
			logger.error(f"Qwen-VL PDF processing failed for {pdf_path} page {page_number}: {e}")
			raise

	def is_available(self) -> bool:
		"""Check if Ollama server is running with Qwen-VL model."""
		try:
			return self.client.is_available()
		except Exception:
			return False

	def supported_languages(self) -> list[str]:
		"""Return list of supported languages (VLMs are multilingual)."""
		return [
			'eng', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'pol', 'rus',
			'jpn', 'kor', 'chi_sim', 'chi_tra', 'ara', 'hin', 'tha', 'vie',
			'ces', 'slk', 'ukr', 'bul', 'ron', 'hun', 'fin', 'swe', 'dan', 'nor'
		]

	def classify_document(
		self,
		image_path: Path,
		categories: list[str] | None = None
	) -> dict:
		"""Classify a document image."""
		result = self.client.classify_document(image_path, categories)

		if categories:
			return {
				'category': result.text.strip(),
				'confidence': 0.9,
				'model': result.model
			}

		parts = result.text.strip().split('|')
		if len(parts) >= 3:
			return {
				'type': parts[0].strip(),
				'category': parts[1].strip(),
				'confidence': parts[2].strip().lower(),
				'model': result.model
			}
		return {
			'raw': result.text.strip(),
			'model': result.model
		}

	def extract_fields(
		self,
		image_path: Path,
		fields: list[str]
	) -> dict[str, str]:
		"""Extract specific fields from a document."""
		result = self.client.extract_fields(image_path, fields)

		extracted = {}
		for line in result.text.strip().split('\n'):
			if ':' in line:
				key, value = line.split(':', 1)
				key = key.strip()
				value = value.strip()
				if value.upper() != 'NOT FOUND':
					extracted[key] = value

		return extracted
