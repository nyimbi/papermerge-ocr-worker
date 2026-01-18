# (c) Copyright Datacraft, 2026
"""VLM prompts for document boundary detection and segmentation."""

# Prompt to detect if multiple documents exist in an image
MULTI_DOCUMENT_DETECTION_PROMPT = """Analyze this scanned image carefully.

TASK: Determine if this image contains MULTIPLE SEPARATE DOCUMENTS.

Signs of multiple documents:
- Clear visual separation (gap, line, or edge) between document areas
- Different paper sizes or orientations
- Multiple letterheads or headers
- Different document types visible (e.g., invoice next to receipt)
- Visible paper edges indicating overlapping or adjacent documents
- Different text orientations or fonts suggesting separate documents

Respond in this EXACT format:
MULTIPLE_DOCUMENTS: YES or NO
COUNT: <number of documents detected, minimum 1>
CONFIDENCE: <0.0 to 1.0>
REASON: <brief explanation>

Example responses:
MULTIPLE_DOCUMENTS: YES
COUNT: 2
CONFIDENCE: 0.9
REASON: Two invoices placed side by side with clear vertical separation

MULTIPLE_DOCUMENTS: NO
COUNT: 1
CONFIDENCE: 0.95
REASON: Single document visible, consistent formatting throughout"""


# Prompt to get boundary coordinates for each document
BOUNDARY_DETECTION_PROMPT = """Analyze this scanned image containing {doc_count} separate documents.

TASK: Identify the EXACT BOUNDARIES of each document in the image.

For each document, provide:
1. Document number (1, 2, 3, etc. from left-to-right, top-to-bottom)
2. Bounding box coordinates as percentages of image dimensions:
   - X: left edge (0% = left side of image, 100% = right side)
   - Y: top edge (0% = top of image, 100% = bottom)
   - WIDTH: width as percentage of image width
   - HEIGHT: height as percentage of image height
3. Rotation angle if document is tilted (positive = clockwise, in degrees)
4. Document type guess (invoice, receipt, letter, form, etc.)
5. Confidence (0.0 to 1.0)

Respond in this EXACT format for each document:

---DOCUMENT 1---
X: <percentage>
Y: <percentage>
WIDTH: <percentage>
HEIGHT: <percentage>
ROTATION: <degrees>
TYPE: <document type>
CONFIDENCE: <0.0-1.0>

---DOCUMENT 2---
X: <percentage>
Y: <percentage>
WIDTH: <percentage>
HEIGHT: <percentage>
ROTATION: <degrees>
TYPE: <document type>
CONFIDENCE: <0.0-1.0>

(continue for all {doc_count} documents)"""


# Prompt for single document skew detection
SKEW_DETECTION_PROMPT = """Analyze this document image.

TASK: Detect if the document is skewed/rotated.

Provide:
1. SKEW_ANGLE: The rotation angle in degrees (positive = clockwise, negative = counter-clockwise)
2. CONFIDENCE: How confident you are (0.0 to 1.0)
3. CORRECTION_NEEDED: YES if angle > 1 degree, NO otherwise

Respond in this EXACT format:
SKEW_ANGLE: <degrees, e.g., 2.5 or -1.3>
CONFIDENCE: <0.0 to 1.0>
CORRECTION_NEEDED: YES or NO"""


# Prompt for document type classification during segmentation
SEGMENT_CLASSIFICATION_PROMPT = """Analyze this document segment.

TASK: Classify the document type and assess quality.

Provide:
1. DOCUMENT_TYPE: (invoice, receipt, contract, letter, form, report, bank_statement, id_document, handwritten_note, technical_drawing, other)
2. LANGUAGE: Primary language of the document
3. QUALITY: (excellent, good, acceptable, poor)
4. READABLE: YES if text is clearly legible, NO otherwise
5. COMPLETE: YES if document appears complete, NO if appears cut off

Respond in this EXACT format:
DOCUMENT_TYPE: <type>
LANGUAGE: <language code, e.g., eng, deu, fra>
QUALITY: <quality level>
READABLE: YES or NO
COMPLETE: YES or NO"""


def get_multi_document_detection_prompt() -> str:
	"""Get prompt for detecting multiple documents."""
	return MULTI_DOCUMENT_DETECTION_PROMPT


def get_boundary_detection_prompt(doc_count: int) -> str:
	"""Get prompt for boundary detection with document count."""
	return BOUNDARY_DETECTION_PROMPT.format(doc_count=doc_count)


def get_skew_detection_prompt() -> str:
	"""Get prompt for skew detection."""
	return SKEW_DETECTION_PROMPT


def get_segment_classification_prompt() -> str:
	"""Get prompt for segment classification."""
	return SEGMENT_CLASSIFICATION_PROMPT
