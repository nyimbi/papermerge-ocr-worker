import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..vlm.client import OllamaClient
from .prompts import CLASSIFICATION_PROMPTS

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
	"""Common document types."""
	UNKNOWN = 'unknown'
	INVOICE = 'invoice'
	RECEIPT = 'receipt'
	CONTRACT = 'contract'
	LETTER = 'letter'
	FORM = 'form'
	REPORT = 'report'
	BANK_STATEMENT = 'bank_statement'
	TAX_FORM = 'tax_form'
	INSURANCE_CLAIM = 'insurance_claim'
	PURCHASE_ORDER = 'purchase_order'
	MEDICAL_RECORD = 'medical_record'
	LEGAL_FILING = 'legal_filing'
	ID_DOCUMENT = 'id_document'
	CORRESPONDENCE = 'correspondence'
	TECHNICAL = 'technical'


class DocumentCategory(str, Enum):
	"""Document categories."""
	UNKNOWN = 'unknown'
	FINANCIAL = 'financial'
	LEGAL = 'legal'
	MEDICAL = 'medical'
	CORRESPONDENCE = 'correspondence'
	ADMINISTRATIVE = 'administrative'
	TECHNICAL = 'technical'
	PERSONAL = 'personal'
	GOVERNMENT = 'government'


class ConfidenceLevel(str, Enum):
	"""Classification confidence levels."""
	HIGH = 'high'
	MEDIUM = 'medium'
	LOW = 'low'


@dataclass
class ClassificationResult:
	"""Result of document classification."""
	document_type: DocumentType
	category: DocumentCategory
	confidence: ConfidenceLevel
	raw_type: str = ''
	raw_category: str = ''
	metadata: dict[str, Any] = field(default_factory=dict)
	processing_time_ms: float = 0

	def to_dict(self) -> dict:
		return {
			'document_type': self.document_type.value,
			'category': self.category.value,
			'confidence': self.confidence.value,
			'raw_type': self.raw_type,
			'raw_category': self.raw_category,
			'metadata': self.metadata,
			'processing_time_ms': self.processing_time_ms
		}


@dataclass
class QualityAssessment:
	"""Document scan quality assessment."""
	readability: str
	alignment: str
	completeness: str
	contrast: str
	issues: list[str]
	overall_quality: str

	def to_dict(self) -> dict:
		return {
			'readability': self.readability,
			'alignment': self.alignment,
			'completeness': self.completeness,
			'contrast': self.contrast,
			'issues': self.issues,
			'overall_quality': self.overall_quality
		}


class DocumentClassifier:
	"""Document type classifier using VLM."""

	TYPE_MAPPING = {
		'invoice': DocumentType.INVOICE,
		'receipt': DocumentType.RECEIPT,
		'contract': DocumentType.CONTRACT,
		'letter': DocumentType.LETTER,
		'business_letter': DocumentType.CORRESPONDENCE,
		'personal_letter': DocumentType.CORRESPONDENCE,
		'form': DocumentType.FORM,
		'application_form': DocumentType.FORM,
		'report': DocumentType.REPORT,
		'bank_statement': DocumentType.BANK_STATEMENT,
		'tax_form': DocumentType.TAX_FORM,
		'insurance_claim': DocumentType.INSURANCE_CLAIM,
		'purchase_order': DocumentType.PURCHASE_ORDER,
		'medical_record': DocumentType.MEDICAL_RECORD,
		'legal_filing': DocumentType.LEGAL_FILING,
		'court_filing': DocumentType.LEGAL_FILING,
		'passport': DocumentType.ID_DOCUMENT,
		'drivers_license': DocumentType.ID_DOCUMENT,
		'national_id': DocumentType.ID_DOCUMENT,
		'memo': DocumentType.CORRESPONDENCE,
		'email': DocumentType.CORRESPONDENCE,
		'email_printout': DocumentType.CORRESPONDENCE,
	}

	CATEGORY_MAPPING = {
		'financial': DocumentCategory.FINANCIAL,
		'legal': DocumentCategory.LEGAL,
		'medical': DocumentCategory.MEDICAL,
		'correspondence': DocumentCategory.CORRESPONDENCE,
		'administrative': DocumentCategory.ADMINISTRATIVE,
		'technical': DocumentCategory.TECHNICAL,
		'personal': DocumentCategory.PERSONAL,
		'government': DocumentCategory.GOVERNMENT,
	}

	def __init__(
		self,
		ollama_client: OllamaClient | None = None,
		base_url: str = 'http://localhost:11434',
		model: str = 'qwen2.5-vl:7b'
	):
		if ollama_client:
			self._client = ollama_client
		else:
			self._client = OllamaClient(base_url=base_url, model=model)

	def classify(
		self,
		image_path: Path,
		domain: str | None = None
	) -> ClassificationResult:
		"""
		Classify a document image.

		Args:
			image_path: Path to document image
			domain: Optional domain hint (financial, legal, medical, etc.)

		Returns:
			ClassificationResult with type, category, and confidence
		"""
		import time
		start_time = time.time()

		prompt_key = domain if domain in CLASSIFICATION_PROMPTS else 'general'
		prompt = CLASSIFICATION_PROMPTS[prompt_key]

		try:
			result = self._client.generate_with_image(
				prompt,
				image_path=image_path,
				temperature=0.0
			)

			response = result.text.strip()
			parsed = self._parse_classification(response)

			processing_time = (time.time() - start_time) * 1000

			return ClassificationResult(
				document_type=parsed['type'],
				category=parsed['category'],
				confidence=parsed['confidence'],
				raw_type=parsed.get('raw_type', ''),
				raw_category=parsed.get('raw_category', ''),
				metadata={'model': result.model, 'prompt_domain': prompt_key},
				processing_time_ms=processing_time
			)

		except Exception as e:
			logger.error(f"Classification failed for {image_path}: {e}")
			return ClassificationResult(
				document_type=DocumentType.UNKNOWN,
				category=DocumentCategory.UNKNOWN,
				confidence=ConfidenceLevel.LOW,
				metadata={'error': str(e)},
				processing_time_ms=(time.time() - start_time) * 1000
			)

	def _parse_classification(self, response: str) -> dict:
		"""Parse VLM classification response."""
		lines = response.strip().split('\n')

		raw_type = ''
		raw_category = ''
		raw_confidence = ''

		for line in lines:
			line = line.strip()
			if line.startswith('TYPE:'):
				raw_type = line.split(':', 1)[1].strip().lower()
			elif line.startswith('CATEGORY:'):
				raw_category = line.split(':', 1)[1].strip().lower()
			elif line.startswith('CONFIDENCE:'):
				raw_confidence = line.split(':', 1)[1].strip().lower()

		if not raw_type and len(lines) == 1:
			raw_type = lines[0].strip().lower()

		doc_type = self.TYPE_MAPPING.get(raw_type, DocumentType.UNKNOWN)
		category = self.CATEGORY_MAPPING.get(raw_category, DocumentCategory.UNKNOWN)

		if raw_confidence in ('high', 'medium', 'low'):
			confidence = ConfidenceLevel(raw_confidence)
		else:
			confidence = ConfidenceLevel.MEDIUM

		return {
			'type': doc_type,
			'category': category,
			'confidence': confidence,
			'raw_type': raw_type,
			'raw_category': raw_category
		}

	def extract_metadata(self, image_path: Path) -> dict[str, str]:
		"""
		Extract key metadata from a document.

		Args:
			image_path: Path to document image

		Returns:
			Dictionary with extracted metadata fields
		"""
		prompt = CLASSIFICATION_PROMPTS['extract_metadata']

		try:
			result = self._client.generate_with_image(
				prompt,
				image_path=image_path,
				temperature=0.0
			)

			return self._parse_metadata(result.text)

		except Exception as e:
			logger.error(f"Metadata extraction failed for {image_path}: {e}")
			return {}

	def _parse_metadata(self, response: str) -> dict[str, str]:
		"""Parse metadata extraction response."""
		metadata = {}

		field_map = {
			'DATE': 'document_date',
			'NUMBER': 'document_number',
			'SENDER': 'sender',
			'RECIPIENT': 'recipient',
			'AMOUNT': 'amount',
			'CURRENCY': 'currency'
		}

		for line in response.strip().split('\n'):
			for key, field_name in field_map.items():
				if line.startswith(f'{key}:'):
					value = line.split(':', 1)[1].strip()
					if value.upper() != 'NOT_FOUND':
						metadata[field_name] = value
					break

		return metadata

	def assess_quality(self, image_path: Path) -> QualityAssessment:
		"""
		Assess the quality of a scanned document.

		Args:
			image_path: Path to document image

		Returns:
			QualityAssessment with various quality metrics
		"""
		prompt = CLASSIFICATION_PROMPTS['quality_check']

		try:
			result = self._client.generate_with_image(
				prompt,
				image_path=image_path,
				temperature=0.0
			)

			return self._parse_quality(result.text)

		except Exception as e:
			logger.error(f"Quality assessment failed for {image_path}: {e}")
			return QualityAssessment(
				readability='unknown',
				alignment='unknown',
				completeness='unknown',
				contrast='unknown',
				issues=['assessment_failed'],
				overall_quality='unknown'
			)

	def _parse_quality(self, response: str) -> QualityAssessment:
		"""Parse quality assessment response."""
		readability = 'unknown'
		alignment = 'unknown'
		completeness = 'unknown'
		contrast = 'unknown'
		issues: list[str] = []

		for line in response.strip().split('\n'):
			line = line.strip()
			if line.startswith('READABILITY:'):
				readability = line.split(':', 1)[1].strip().lower()
			elif line.startswith('ALIGNMENT:'):
				alignment = line.split(':', 1)[1].strip().lower()
			elif line.startswith('COMPLETENESS:'):
				completeness = line.split(':', 1)[1].strip().lower()
			elif line.startswith('CONTRAST:'):
				contrast = line.split(':', 1)[1].strip().lower()
			elif line.startswith('ISSUES:'):
				issues_str = line.split(':', 1)[1].strip()
				if issues_str.upper() != 'NONE':
					issues = [i.strip() for i in issues_str.split(',')]

		quality_scores = {
			'good': 3,
			'fair': 2,
			'poor': 1,
			'complete': 3,
			'partial': 2,
			'unclear': 1,
			'slight_skew': 2,
			'significant_skew': 1
		}

		scores = [
			quality_scores.get(readability, 2),
			quality_scores.get(alignment, 2),
			quality_scores.get(completeness, 2),
			quality_scores.get(contrast, 2)
		]
		avg_score = sum(scores) / len(scores)

		if avg_score >= 2.5:
			overall = 'good'
		elif avg_score >= 1.5:
			overall = 'fair'
		else:
			overall = 'poor'

		return QualityAssessment(
			readability=readability,
			alignment=alignment,
			completeness=completeness,
			contrast=contrast,
			issues=issues,
			overall_quality=overall
		)
