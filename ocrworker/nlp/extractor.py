# (c) Copyright Datacraft, 2026
"""SpaCy-based metadata extraction from document text."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .patterns import PatternMatcher, get_patterns_for_document_type

logger = logging.getLogger(__name__)


@dataclass
class Entity:
	"""Extracted entity."""
	text: str
	label: str
	start: int
	end: int
	confidence: float = 1.0
	normalized_value: Any = None
	source: str = 'spacy'  # spacy, pattern, or custom


@dataclass
class ExtractedMetadata:
	"""Complete extracted metadata from a document."""
	# Standard NER entities
	persons: list[str] = field(default_factory=list)
	organizations: list[str] = field(default_factory=list)
	locations: list[str] = field(default_factory=list)
	dates: list[str] = field(default_factory=list)
	money: list[str] = field(default_factory=list)

	# Document-specific fields
	document_date: str | None = None
	due_date: str | None = None
	invoice_number: str | None = None
	po_number: str | None = None
	order_number: str | None = None
	account_number: str | None = None
	reference_number: str | None = None

	# Financial
	total_amount: float | None = None
	subtotal: float | None = None
	tax_amount: float | None = None
	currency: str | None = None

	# Parties
	vendor_name: str | None = None
	client_name: str | None = None
	signatory: str | None = None
	contact_name: str | None = None

	# Contact info
	emails: list[str] = field(default_factory=list)
	phones: list[str] = field(default_factory=list)
	urls: list[str] = field(default_factory=list)

	# Tax identifiers
	tax_ids: list[str] = field(default_factory=list)
	vat_numbers: list[str] = field(default_factory=list)

	# All entities for custom processing
	all_entities: list[Entity] = field(default_factory=list)

	# Processing metadata
	language: str = 'en'
	processing_time_ms: float = 0
	model_used: str = ''

	def to_dict(self) -> dict:
		"""Convert to dictionary for storage."""
		return {
			'persons': self.persons,
			'organizations': self.organizations,
			'locations': self.locations,
			'dates': self.dates,
			'money': self.money,
			'document_date': self.document_date,
			'due_date': self.due_date,
			'invoice_number': self.invoice_number,
			'po_number': self.po_number,
			'order_number': self.order_number,
			'account_number': self.account_number,
			'reference_number': self.reference_number,
			'total_amount': self.total_amount,
			'subtotal': self.subtotal,
			'tax_amount': self.tax_amount,
			'currency': self.currency,
			'vendor_name': self.vendor_name,
			'client_name': self.client_name,
			'signatory': self.signatory,
			'emails': self.emails,
			'phones': self.phones,
			'tax_ids': self.tax_ids,
			'vat_numbers': self.vat_numbers,
			'language': self.language,
		}


class MetadataExtractor:
	"""
	Extract structured metadata from document text using SpaCy NLP.

	Combines:
	- SpaCy NER for persons, organizations, locations, dates, money
	- Custom pattern matching for document-specific entities
	- Context-aware field assignment
	"""

	# SpaCy model mapping
	MODEL_MAP = {
		'en': 'en_core_web_trf',  # English transformer model
		'de': 'de_core_news_lg',
		'fr': 'fr_core_news_lg',
		'es': 'es_core_news_lg',
		'it': 'it_core_news_lg',
		'nl': 'nl_core_news_lg',
		'pt': 'pt_core_news_lg',
	}

	# Fallback to smaller models
	FALLBACK_MODELS = {
		'en': 'en_core_web_lg',
	}

	def __init__(
		self,
		language: str = 'en',
		model_name: str | None = None,
		use_gpu: bool = False,
	):
		self.language = language
		self._nlp = None
		self._model_name = model_name or self.MODEL_MAP.get(language, 'en_core_web_trf')
		self._use_gpu = use_gpu
		self._pattern_matcher = PatternMatcher()
		self._loaded = False

	def _load_model(self):
		"""Load SpaCy model lazily."""
		if self._loaded:
			return

		import spacy

		if self._use_gpu:
			spacy.require_gpu()

		try:
			self._nlp = spacy.load(self._model_name)
			logger.info(f"Loaded SpaCy model: {self._model_name}")
		except OSError:
			# Try fallback model
			fallback = self.FALLBACK_MODELS.get(self.language)
			if fallback:
				try:
					self._nlp = spacy.load(fallback)
					self._model_name = fallback
					logger.info(f"Loaded fallback SpaCy model: {fallback}")
				except OSError:
					logger.warning(f"SpaCy model not found, using blank: {self.language}")
					self._nlp = spacy.blank(self.language)
			else:
				logger.warning(f"SpaCy model not found, using blank: {self.language}")
				self._nlp = spacy.blank(self.language)

		self._loaded = True

	def extract(
		self,
		text: str,
		document_type: str | None = None,
	) -> ExtractedMetadata:
		"""
		Extract metadata from document text.

		Args:
			text: Document text content
			document_type: Optional document type hint (invoice, contract, etc.)

		Returns:
			ExtractedMetadata with all extracted information
		"""
		import time
		start_time = time.time()

		self._load_model()

		metadata = ExtractedMetadata(
			language=self.language,
			model_used=self._model_name,
		)

		if not text or not text.strip():
			return metadata

		# Process with SpaCy
		doc = self._nlp(text)

		# Extract SpaCy NER entities
		for ent in doc.ents:
			entity = Entity(
				text=ent.text,
				label=ent.label_,
				start=ent.start_char,
				end=ent.end_char,
				source='spacy',
			)
			metadata.all_entities.append(entity)

			# Categorize by type
			if ent.label_ == 'PERSON':
				if ent.text not in metadata.persons:
					metadata.persons.append(ent.text)
			elif ent.label_ == 'ORG':
				if ent.text not in metadata.organizations:
					metadata.organizations.append(ent.text)
			elif ent.label_ in ('GPE', 'LOC', 'FAC'):
				if ent.text not in metadata.locations:
					metadata.locations.append(ent.text)
			elif ent.label_ == 'DATE':
				if ent.text not in metadata.dates:
					metadata.dates.append(ent.text)
			elif ent.label_ == 'MONEY':
				if ent.text not in metadata.money:
					metadata.money.append(ent.text)

		# Apply pattern matching
		pattern_matches = self._pattern_matcher.match(text)

		# Add document-type-specific patterns
		if document_type:
			doc_patterns = get_patterns_for_document_type(document_type)
			for pattern in doc_patterns:
				self._pattern_matcher.add_rule(pattern)
			pattern_matches.extend(self._pattern_matcher.match(text))

		# Process pattern matches
		for match in pattern_matches:
			entity = Entity(
				text=match.text,
				label=match.entity_type,
				start=match.start,
				end=match.end,
				confidence=match.confidence,
				normalized_value=match.normalized_value,
				source='pattern',
			)
			metadata.all_entities.append(entity)

			# Assign to specific fields
			self._assign_field(metadata, match)

		# Post-processing: assign context-aware fields
		self._assign_contextual_fields(metadata, text)

		metadata.processing_time_ms = (time.time() - start_time) * 1000

		return metadata

	def _assign_field(self, metadata: ExtractedMetadata, match):
		"""Assign pattern match to specific metadata field."""
		entity_type = match.entity_type

		if entity_type == 'INVOICE_NUMBER' and not metadata.invoice_number:
			metadata.invoice_number = match.text

		elif entity_type == 'PO_NUMBER' and not metadata.po_number:
			metadata.po_number = match.text

		elif entity_type == 'ORDER_NUMBER' and not metadata.order_number:
			metadata.order_number = match.text

		elif entity_type == 'ACCOUNT_NUMBER' and not metadata.account_number:
			metadata.account_number = match.text

		elif entity_type == 'REFERENCE_NUMBER' and not metadata.reference_number:
			metadata.reference_number = match.text

		elif entity_type in ('TOTAL_AMOUNT',) and not metadata.total_amount:
			if match.normalized_value is not None:
				metadata.total_amount = match.normalized_value

		elif entity_type == 'SUBTOTAL' and not metadata.subtotal:
			if match.normalized_value is not None:
				metadata.subtotal = match.normalized_value

		elif entity_type == 'TAX_AMOUNT' and not metadata.tax_amount:
			if match.normalized_value is not None:
				metadata.tax_amount = match.normalized_value

		elif entity_type in ('MONEY_USD', 'MONEY_EUR', 'MONEY_GBP'):
			if not metadata.currency:
				metadata.currency = entity_type.split('_')[1]
			if match.normalized_value is not None:
				metadata.money.append(str(match.normalized_value))

		elif entity_type in ('INVOICE_DATE', 'DATE') and not metadata.document_date:
			metadata.document_date = match.text
			if match.text not in metadata.dates:
				metadata.dates.append(match.text)

		elif entity_type == 'DUE_DATE' and not metadata.due_date:
			metadata.due_date = match.text

		elif entity_type == 'EMAIL':
			if match.text not in metadata.emails:
				metadata.emails.append(match.text)

		elif entity_type == 'PHONE':
			if match.text not in metadata.phones:
				metadata.phones.append(match.text)

		elif entity_type == 'URL':
			if match.text not in metadata.urls:
				metadata.urls.append(match.text)

		elif entity_type in ('TAX_ID_EIN', 'SSN'):
			if match.text not in metadata.tax_ids:
				metadata.tax_ids.append(match.text)

		elif entity_type == 'VAT_NUMBER':
			if match.text not in metadata.vat_numbers:
				metadata.vat_numbers.append(match.text)

	def _assign_contextual_fields(self, metadata: ExtractedMetadata, text: str):
		"""Assign context-aware fields based on document structure."""
		text_lower = text.lower()

		# Try to identify vendor (usually first org mentioned after "from", "seller", etc.)
		if metadata.organizations and not metadata.vendor_name:
			for org in metadata.organizations[:3]:
				org_lower = org.lower()
				idx = text_lower.find(org_lower)
				if idx > 0:
					context_before = text_lower[max(0, idx - 50):idx]
					if any(w in context_before for w in ['from', 'seller', 'vendor', 'bill from']):
						metadata.vendor_name = org
						break

		# Try to identify client (usually after "to", "bill to", "customer", etc.)
		if metadata.organizations and not metadata.client_name:
			for org in metadata.organizations:
				if org == metadata.vendor_name:
					continue
				org_lower = org.lower()
				idx = text_lower.find(org_lower)
				if idx > 0:
					context_before = text_lower[max(0, idx - 50):idx]
					if any(w in context_before for w in ['to', 'bill to', 'customer', 'client', 'ship to']):
						metadata.client_name = org
						break

		# Try to identify signatory from person names
		if metadata.persons and not metadata.signatory:
			for person in metadata.persons:
				person_lower = person.lower()
				idx = text_lower.find(person_lower)
				if idx > 0:
					context_before = text_lower[max(0, idx - 100):idx]
					context_after = text_lower[idx:min(len(text_lower), idx + 100)]
					if any(w in context_before + context_after for w in
					       ['signature', 'signed by', 'authorized', 'behalf']):
						metadata.signatory = person
						break

		# Contact name
		if metadata.persons and not metadata.contact_name:
			for person in metadata.persons:
				if person == metadata.signatory:
					continue
				person_lower = person.lower()
				idx = text_lower.find(person_lower)
				if idx > 0:
					context = text_lower[max(0, idx - 50):min(len(text_lower), idx + 50)]
					if any(w in context for w in ['contact', 'phone', 'email', 'attn']):
						metadata.contact_name = person
						break

	async def extract_async(
		self,
		text: str,
		document_type: str | None = None,
	) -> ExtractedMetadata:
		"""Async wrapper for extraction."""
		import asyncio
		return await asyncio.to_thread(self.extract, text, document_type)

	@classmethod
	def get_available_languages(cls) -> list[str]:
		"""Get list of languages with available models."""
		return list(cls.MODEL_MAP.keys())

	@classmethod
	def download_model(cls, language: str) -> bool:
		"""Download SpaCy model for language."""
		model_name = cls.MODEL_MAP.get(language)
		if not model_name:
			return False

		try:
			import subprocess
			result = subprocess.run(
				['python', '-m', 'spacy', 'download', model_name],
				capture_output=True,
				text=True,
			)
			return result.returncode == 0
		except Exception as e:
			logger.error(f"Failed to download model {model_name}: {e}")
			return False
