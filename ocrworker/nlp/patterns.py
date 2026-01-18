# (c) Copyright Datacraft, 2026
"""Pattern matching for document-specific entities."""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PatternRule:
	"""Rule for pattern-based entity extraction."""
	name: str
	pattern: str  # Regex pattern
	entity_type: str  # Entity label
	priority: int = 0
	validator: Callable[[str], bool] | None = None
	normalizer: Callable[[str], str] | None = None
	context_words: list[str] = field(default_factory=list)  # Words that should appear nearby
	flags: int = re.IGNORECASE


@dataclass
class MatchResult:
	"""Result of a pattern match."""
	text: str
	entity_type: str
	start: int
	end: int
	pattern_name: str
	normalized_value: Any = None
	confidence: float = 1.0


class PatternMatcher:
	"""
	Pattern-based entity matcher for document-specific patterns.

	Extracts entities like invoice numbers, dates, amounts that
	may not be recognized by standard NER models.
	"""

	def __init__(self):
		self._rules: list[PatternRule] = []
		self._compiled: dict[str, re.Pattern] = {}
		self._setup_default_rules()

	def _setup_default_rules(self):
		"""Set up default pattern rules for common document entities."""

		# Invoice/PO Numbers
		self.add_rule(PatternRule(
			name='invoice_number',
			pattern=r'\b(?:INV|INVOICE)[-#\s]*(\d{4,12})\b',
			entity_type='INVOICE_NUMBER',
			context_words=['invoice', 'bill', 'statement'],
		))

		self.add_rule(PatternRule(
			name='po_number',
			pattern=r'\b(?:PO|P\.O\.|PURCHASE\s*ORDER)[-#\s]*(\d{4,12})\b',
			entity_type='PO_NUMBER',
			context_words=['purchase', 'order', 'po'],
		))

		self.add_rule(PatternRule(
			name='order_number',
			pattern=r'\b(?:ORDER|ORD)[-#\s]*(\d{4,12})\b',
			entity_type='ORDER_NUMBER',
		))

		# Account/Reference Numbers
		self.add_rule(PatternRule(
			name='account_number',
			pattern=r'\b(?:ACCT?|ACCOUNT)[-#\s]*(\d{6,16})\b',
			entity_type='ACCOUNT_NUMBER',
			context_words=['account', 'acct'],
		))

		self.add_rule(PatternRule(
			name='reference_number',
			pattern=r'\b(?:REF|REFERENCE)[-#\s]*([A-Z0-9]{4,20})\b',
			entity_type='REFERENCE_NUMBER',
		))

		# Monetary Values
		self.add_rule(PatternRule(
			name='usd_amount',
			pattern=r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
			entity_type='MONEY_USD',
			normalizer=self._normalize_money,
		))

		self.add_rule(PatternRule(
			name='eur_amount',
			pattern=r'€\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
			entity_type='MONEY_EUR',
			normalizer=self._normalize_money_eu,
		))

		self.add_rule(PatternRule(
			name='gbp_amount',
			pattern=r'£\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
			entity_type='MONEY_GBP',
			normalizer=self._normalize_money,
		))

		self.add_rule(PatternRule(
			name='generic_amount',
			pattern=r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2}))\s*(?:USD|EUR|GBP|CAD|AUD)\b',
			entity_type='MONEY',
			normalizer=self._normalize_money,
		))

		# Dates in various formats
		self.add_rule(PatternRule(
			name='date_mdy',
			pattern=r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',
			entity_type='DATE',
			validator=self._validate_date_mdy,
			normalizer=self._normalize_date_mdy,
		))

		self.add_rule(PatternRule(
			name='date_dmy',
			pattern=r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',
			entity_type='DATE',
			priority=-1,  # Lower priority than MDY
			validator=self._validate_date_dmy,
		))

		self.add_rule(PatternRule(
			name='date_iso',
			pattern=r'\b(\d{4})-(\d{2})-(\d{2})\b',
			entity_type='DATE',
			priority=1,  # Higher priority for ISO format
			normalizer=lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
		))

		self.add_rule(PatternRule(
			name='date_written',
			pattern=r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
			        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
			        r'Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
			entity_type='DATE',
			normalizer=self._normalize_date_written,
		))

		# Tax IDs
		self.add_rule(PatternRule(
			name='ein',
			pattern=r'\b(\d{2})-(\d{7})\b',
			entity_type='TAX_ID_EIN',
			context_words=['ein', 'tax', 'employer'],
			validator=lambda m: True,  # Basic format validation
		))

		self.add_rule(PatternRule(
			name='ssn',
			pattern=r'\b(\d{3})-(\d{2})-(\d{4})\b',
			entity_type='SSN',
			context_words=['ssn', 'social', 'security'],
		))

		self.add_rule(PatternRule(
			name='vat_eu',
			pattern=r'\b([A-Z]{2})(\d{8,12})\b',
			entity_type='VAT_NUMBER',
			context_words=['vat', 'btw', 'mwst', 'iva'],
		))

		# Phone Numbers
		self.add_rule(PatternRule(
			name='phone_us',
			pattern=r'\b(?:\+1\s*)?(?:\(\d{3}\)|\d{3})[-.\s]*\d{3}[-.\s]*\d{4}\b',
			entity_type='PHONE',
		))

		self.add_rule(PatternRule(
			name='phone_intl',
			pattern=r'\b\+\d{1,3}[-.\s]*\d{1,4}[-.\s]*\d{4,10}\b',
			entity_type='PHONE',
		))

		# Email
		self.add_rule(PatternRule(
			name='email',
			pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
			entity_type='EMAIL',
		))

		# URLs
		self.add_rule(PatternRule(
			name='url',
			pattern=r'\bhttps?://[^\s<>\[\]]+\b',
			entity_type='URL',
		))

		# Percentages
		self.add_rule(PatternRule(
			name='percentage',
			pattern=r'\b(\d+(?:\.\d+)?)\s*%',
			entity_type='PERCENTAGE',
			normalizer=lambda m: float(m.group(1)),
		))

	def add_rule(self, rule: PatternRule):
		"""Add a pattern rule."""
		self._rules.append(rule)
		self._rules.sort(key=lambda r: -r.priority)
		self._compiled[rule.name] = re.compile(rule.pattern, rule.flags)

	def match(self, text: str) -> list[MatchResult]:
		"""Find all pattern matches in text."""
		results = []
		seen_spans = set()

		for rule in self._rules:
			pattern = self._compiled.get(rule.name)
			if not pattern:
				continue

			for match in pattern.finditer(text):
				span = (match.start(), match.end())

				# Skip overlapping matches
				if any(
					s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1]
					for s in seen_spans
				):
					continue

				# Validate if validator provided
				if rule.validator and not rule.validator(match.group(0)):
					continue

				# Check context words if provided
				confidence = 1.0
				if rule.context_words:
					context_start = max(0, match.start() - 100)
					context_end = min(len(text), match.end() + 100)
					context = text[context_start:context_end].lower()

					found_context = any(
						word in context for word in rule.context_words
					)
					if found_context:
						confidence = 1.0
					else:
						confidence = 0.7

				# Normalize value if normalizer provided
				normalized = None
				if rule.normalizer:
					try:
						normalized = rule.normalizer(match)
					except Exception as e:
						logger.debug(f"Normalization failed for {rule.name}: {e}")

				result = MatchResult(
					text=match.group(0),
					entity_type=rule.entity_type,
					start=match.start(),
					end=match.end(),
					pattern_name=rule.name,
					normalized_value=normalized,
					confidence=confidence,
				)
				results.append(result)
				seen_spans.add(span)

		return results

	# Normalizers

	def _normalize_money(self, match) -> float:
		"""Normalize US/UK format money to float."""
		text = match.group(1) if match.lastindex else match.group(0)
		text = text.replace(',', '').replace('$', '').replace('£', '').strip()
		return float(text)

	def _normalize_money_eu(self, match) -> float:
		"""Normalize EU format money to float."""
		text = match.group(1) if match.lastindex else match.group(0)
		text = text.replace('.', '').replace(',', '.').replace('€', '').strip()
		return float(text)

	def _normalize_date_mdy(self, match) -> str:
		"""Normalize M/D/Y to ISO format."""
		month, day, year = match.groups()
		if len(year) == 2:
			year = f"20{year}" if int(year) < 50 else f"19{year}"
		return f"{year}-{int(month):02d}-{int(day):02d}"

	def _normalize_date_written(self, match) -> str:
		"""Normalize written date to ISO format."""
		months = {
			'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
			'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
			'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
		}
		month_str, day, year = match.groups()
		month = months.get(month_str[:3].lower(), '01')
		return f"{year}-{month}-{int(day):02d}"

	# Validators

	def _validate_date_mdy(self, text: str) -> bool:
		"""Validate M/D/Y date format."""
		match = re.match(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', text)
		if not match:
			return False
		month, day, _ = match.groups()
		return 1 <= int(month) <= 12 and 1 <= int(day) <= 31

	def _validate_date_dmy(self, text: str) -> bool:
		"""Validate D/M/Y date format."""
		match = re.match(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', text)
		if not match:
			return False
		day, month, _ = match.groups()
		return 1 <= int(month) <= 12 and 1 <= int(day) <= 31


# Document type specific patterns
INVOICE_PATTERNS = [
	PatternRule(
		name='invoice_total',
		pattern=r'(?:TOTAL|AMOUNT\s*DUE|GRAND\s*TOTAL)\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
		entity_type='TOTAL_AMOUNT',
		context_words=['total', 'due', 'pay'],
	),
	PatternRule(
		name='invoice_subtotal',
		pattern=r'(?:SUBTOTAL|SUB-TOTAL)\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
		entity_type='SUBTOTAL',
	),
	PatternRule(
		name='invoice_tax',
		pattern=r'(?:TAX|VAT|GST|HST)\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
		entity_type='TAX_AMOUNT',
	),
	PatternRule(
		name='due_date',
		pattern=r'(?:DUE\s*DATE|PAYMENT\s*DUE|DUE\s*BY)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
		entity_type='DUE_DATE',
	),
	PatternRule(
		name='invoice_date',
		pattern=r'(?:INVOICE\s*DATE|DATE\s*OF\s*INVOICE|BILL\s*DATE)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
		entity_type='INVOICE_DATE',
	),
]

CONTRACT_PATTERNS = [
	PatternRule(
		name='effective_date',
		pattern=r'(?:EFFECTIVE\s*DATE|COMMENCEMENT\s*DATE)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
		entity_type='EFFECTIVE_DATE',
	),
	PatternRule(
		name='expiration_date',
		pattern=r'(?:EXPIRATION\s*DATE|TERMINATION\s*DATE|END\s*DATE)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
		entity_type='EXPIRATION_DATE',
	),
	PatternRule(
		name='contract_value',
		pattern=r'(?:CONTRACT\s*VALUE|TOTAL\s*VALUE|AGREEMENT\s*AMOUNT)\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
		entity_type='CONTRACT_VALUE',
	),
]


def get_patterns_for_document_type(doc_type: str) -> list[PatternRule]:
	"""Get document-type-specific patterns."""
	patterns_map = {
		'invoice': INVOICE_PATTERNS,
		'contract': CONTRACT_PATTERNS,
	}
	return patterns_map.get(doc_type.lower(), [])
