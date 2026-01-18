# (c) Copyright Datacraft, 2026
"""NLP module for document metadata extraction using SpaCy."""
from .extractor import MetadataExtractor, ExtractedMetadata
from .patterns import PatternMatcher, PatternRule

__all__ = [
	'MetadataExtractor',
	'ExtractedMetadata',
	'PatternMatcher',
	'PatternRule',
]
