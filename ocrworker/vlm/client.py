import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OllamaVLMResult:
	"""Result from Ollama VLM processing."""
	text: str
	model: str
	prompt_eval_count: int = 0
	eval_count: int = 0
	total_duration_ms: float = 0
	load_duration_ms: float = 0
	prompt_eval_duration_ms: float = 0
	eval_duration_ms: float = 0
	metadata: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
	"""HTTP client for Ollama VLM API."""

	DEFAULT_TIMEOUT = 120.0
	DEFAULT_MODEL = 'qwen2.5-vl:7b'

	def __init__(
		self,
		base_url: str = 'http://localhost:11434',
		model: str | None = None,
		timeout: float = DEFAULT_TIMEOUT
	):
		self.base_url = base_url.rstrip('/')
		self.model = model or self.DEFAULT_MODEL
		self.timeout = timeout
		self._client = httpx.Client(timeout=timeout)

	def close(self):
		"""Close the HTTP client."""
		self._client.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

	def is_available(self) -> bool:
		"""Check if Ollama server is running and model is available."""
		try:
			response = self._client.get(f"{self.base_url}/api/tags")
			if response.status_code != 200:
				return False

			models = response.json().get('models', [])
			model_names = [m.get('name', '').split(':')[0] for m in models]
			return self.model.split(':')[0] in model_names
		except Exception as e:
			logger.debug(f"Ollama availability check failed: {e}")
			return False

	def list_models(self) -> list[dict]:
		"""List available models."""
		try:
			response = self._client.get(f"{self.base_url}/api/tags")
			if response.status_code == 200:
				return response.json().get('models', [])
			return []
		except Exception:
			return []

	def pull_model(self, model: str | None = None) -> bool:
		"""Pull a model from Ollama registry."""
		model = model or self.model
		try:
			response = self._client.post(
				f"{self.base_url}/api/pull",
				json={'name': model},
				timeout=600.0
			)
			return response.status_code == 200
		except Exception as e:
			logger.error(f"Failed to pull model {model}: {e}")
			return False

	def generate_with_image(
		self,
		prompt: str,
		image_path: Path | None = None,
		image_base64: str | None = None,
		model: str | None = None,
		temperature: float = 0.1,
		max_tokens: int = 4096,
		**kwargs
	) -> OllamaVLMResult:
		"""
		Generate text from an image using VLM.

		Args:
			prompt: Text prompt for the model
			image_path: Path to image file
			image_base64: Base64-encoded image (alternative to image_path)
			model: Model to use (defaults to instance model)
			temperature: Generation temperature
			max_tokens: Maximum tokens to generate
			**kwargs: Additional generation parameters

		Returns:
			OllamaVLMResult with generated text and metrics
		"""
		start_time = time.time()

		if image_path and not image_base64:
			with open(image_path, 'rb') as f:
				image_base64 = base64.b64encode(f.read()).decode('utf-8')

		if not image_base64:
			raise ValueError("Either image_path or image_base64 must be provided")

		payload = {
			'model': model or self.model,
			'prompt': prompt,
			'images': [image_base64],
			'stream': False,
			'options': {
				'temperature': temperature,
				'num_predict': max_tokens,
				**kwargs
			}
		}

		try:
			response = self._client.post(
				f"{self.base_url}/api/generate",
				json=payload,
				timeout=self.timeout
			)
			response.raise_for_status()
			data = response.json()

			total_duration = (time.time() - start_time) * 1000

			return OllamaVLMResult(
				text=data.get('response', ''),
				model=data.get('model', model or self.model),
				prompt_eval_count=data.get('prompt_eval_count', 0),
				eval_count=data.get('eval_count', 0),
				total_duration_ms=total_duration,
				load_duration_ms=data.get('load_duration', 0) / 1_000_000,
				prompt_eval_duration_ms=data.get('prompt_eval_duration', 0) / 1_000_000,
				eval_duration_ms=data.get('eval_duration', 0) / 1_000_000,
				metadata={'done': data.get('done', False)}
			)

		except httpx.HTTPStatusError as e:
			logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
			raise
		except Exception as e:
			logger.error(f"Ollama request failed: {e}")
			raise

	def ocr_image(
		self,
		image_path: Path,
		lang: str = 'eng',
		detailed: bool = False
	) -> OllamaVLMResult:
		"""
		Perform OCR on an image using VLM.

		Args:
			image_path: Path to image file
			lang: Language hint for OCR
			detailed: If True, request structured output with positions

		Returns:
			OllamaVLMResult with extracted text
		"""
		if detailed:
			prompt = f"""Analyze this document image and extract ALL text content.
For each text region, provide:
1. The exact text content
2. Approximate position (top, middle, bottom of page; left, center, right)
3. Text type (heading, paragraph, table cell, label, value, signature area)

Language hint: {lang}

Format your response as structured text, preserving the document's logical layout.
Include ALL visible text, numbers, dates, and any handwritten content you can read."""
		else:
			prompt = f"""Extract ALL text from this document image.
Preserve the original layout and structure as much as possible.
Include all visible text, numbers, dates, and readable handwritten content.
Language: {lang}"""

		return self.generate_with_image(prompt, image_path=image_path)

	def classify_document(
		self,
		image_path: Path,
		categories: list[str] | None = None
	) -> OllamaVLMResult:
		"""
		Classify a document image.

		Args:
			image_path: Path to image file
			categories: Optional list of category names to choose from

		Returns:
			OllamaVLMResult with classification
		"""
		if categories:
			cat_list = '\n'.join(f"- {c}" for c in categories)
			prompt = f"""Classify this document into ONE of the following categories:
{cat_list}

Respond with ONLY the category name, nothing else."""
		else:
			prompt = """Classify this document. Identify:
1. Document type (invoice, receipt, contract, letter, form, report, etc.)
2. General category (financial, legal, medical, correspondence, administrative, etc.)
3. Confidence level (high, medium, low)

Format: TYPE | CATEGORY | CONFIDENCE"""

		return self.generate_with_image(prompt, image_path=image_path, temperature=0.0)

	def extract_fields(
		self,
		image_path: Path,
		fields: list[str]
	) -> OllamaVLMResult:
		"""
		Extract specific fields from a document.

		Args:
			image_path: Path to image file
			fields: List of field names to extract

		Returns:
			OllamaVLMResult with extracted fields
		"""
		field_list = '\n'.join(f"- {f}" for f in fields)
		prompt = f"""Extract the following fields from this document:
{field_list}

For each field, provide the value found in the document.
If a field is not found, respond with "NOT FOUND".

Format each response as:
FIELD_NAME: value"""

		return self.generate_with_image(prompt, image_path=image_path, temperature=0.0)
