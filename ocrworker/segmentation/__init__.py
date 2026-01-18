# (c) Copyright Datacraft, 2026
"""Multi-document segmentation module.

Detects and splits multiple documents from a single scanned image.
"""
from .segmenter import DocumentSegmenter
from .models import (
	SegmentedDocument,
	DocumentBoundary,
	SegmentationResult,
	SegmentationMethod,
)

__all__ = [
	'DocumentSegmenter',
	'SegmentedDocument',
	'DocumentBoundary',
	'SegmentationResult',
	'SegmentationMethod',
]
