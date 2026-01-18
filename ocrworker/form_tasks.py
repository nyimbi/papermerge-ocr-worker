# (c) Copyright Datacraft, 2026
"""Celery tasks for form recognition and signature extraction."""
import logging
import json
from pathlib import Path
from uuid import UUID, uuid4

from celery import shared_task
from celery.app import default_app as celery_app

from ocrworker import config, plib, s3
from ocrworker.db.engine import Session
from ocrworker import constants as const
from ocrworker.form_detector import FormDetector, FormDetectionResult
from ocrworker.signature_extractor import SignatureExtractor

logger = logging.getLogger(__name__)
settings = config.get_settings()


@shared_task(name="darchiva.form.detect")
def detect_form_task(
	document_id: str,
	document_version_id: str,
	page_ids: list[str],
	tenant_id: str,
) -> dict:
	"""Detect if document is a form and extract fields."""
	logger.info(f"Starting form detection for document {document_id}")

	# Download page images
	image_paths = []
	ocr_results = []

	for page_id in page_ids:
		# Download page image
		page_path = plib.abs_page_path(UUID(page_id))
		s3.download_page_dir(UUID(page_id))

		# Find image file
		image_files = list(page_path.glob("*.png")) + list(page_path.glob("*.jpg"))
		if image_files:
			image_paths.append(image_files[0])

		# Load OCR results
		ocr_file = page_path / "ocr.json"
		if ocr_file.exists():
			with open(ocr_file) as f:
				ocr_results.append(json.load(f))
		else:
			# Try to get from text file
			txt_file = page_path / f"{page_id}.txt"
			if txt_file.exists():
				text = txt_file.read_text()
				ocr_results.append({
					"blocks": [{"text": text, "bbox": {}}]
				})
			else:
				ocr_results.append({"blocks": []})

	# Detect form
	detector = FormDetector()
	result = detector.detect_form(image_paths, ocr_results)

	# Send result to core for storage
	celery_app.send_task(
		"papermerge.core.tasks.store_form_detection",
		kwargs={
			"document_id": document_id,
			"tenant_id": tenant_id,
			"is_form": result.is_form,
			"form_type": result.form_type,
			"fields": [
				{
					"field_type": f.field_type,
					"label": f.label,
					"value": f.value,
					"bbox": f.bbox.to_dict(),
					"confidence": f.confidence,
					"page_number": f.page_number,
				}
				for f in result.fields
			],
			"confidence": result.confidence,
		},
		route_name="default",
	)

	logger.info(f"Form detection complete for {document_id}: is_form={result.is_form}")

	return {
		"document_id": document_id,
		"is_form": result.is_form,
		"form_type": result.form_type,
		"field_count": len(result.fields),
		"signature_count": len(result.signatures),
	}


@shared_task(name="darchiva.signature.extract")
def extract_signatures_task(
	document_id: str,
	document_version_id: str,
	page_ids: list[str],
	signature_regions: list[dict] | None = None,
	tenant_id: str | None = None,
) -> dict:
	"""Extract signatures from document pages."""
	logger.info(f"Starting signature extraction for document {document_id}")

	extractor = SignatureExtractor()
	all_signatures = []

	for page_num, page_id in enumerate(page_ids, 1):
		# Download page
		page_path = plib.abs_page_path(UUID(page_id))
		s3.download_page_dir(UUID(page_id))

		# Find image
		image_files = list(page_path.glob("*.png")) + list(page_path.glob("*.jpg"))
		if not image_files:
			continue

		image_path = image_files[0]

		# Extract signatures
		if signature_regions:
			# Use provided regions
			page_regions = [r for r in signature_regions if r.get("page_number") == page_num]
			signatures = extractor.extract_signatures(image_path, page_regions, page_num)
		else:
			# Auto-detect
			signatures = extractor.extract_all_signatures(image_path, page_num)

		all_signatures.extend(signatures)

	# Store signatures
	for sig in all_signatures:
		# Upload signature image to S3
		sig_key = f"signatures/{document_id}/{sig.id}.png"
		s3.upload_bytes(sig.image_png, sig_key)

		# Send to core for storage
		celery_app.send_task(
			"papermerge.core.tasks.store_signature",
			kwargs={
				"document_id": document_id,
				"tenant_id": tenant_id,
				"signature_id": str(sig.id),
				"page_number": sig.page_number,
				"bounding_box": sig.bounding_box,
				"signature_type": sig.signature_type,
				"confidence": sig.confidence,
				"image_s3_key": sig_key,
				"svg_data": sig.image_svg,
			},
			route_name="default",
		)

	logger.info(f"Extracted {len(all_signatures)} signatures from {document_id}")

	return {
		"document_id": document_id,
		"signature_count": len(all_signatures),
		"signatures": [
			{
				"id": str(s.id),
				"page": s.page_number,
				"type": s.signature_type,
			}
			for s in all_signatures
		],
	}


@shared_task(name="darchiva.form.process")
def process_form_document_task(
	document_id: str,
	template_id: str | None,
	tenant_id: str,
) -> dict:
	"""Full form processing pipeline: detection + extraction + signatures."""
	logger.info(f"Starting form processing pipeline for {document_id}")

	with Session() as db_session:
		from ocrworker.db import api as db
		doc_ver = db.get_last_version(db_session, doc_id=UUID(document_id))
		pages = db.get_pages(db_session, doc_ver_id=doc_ver.id)

	page_ids = [str(p.id) for p in pages]

	# Step 1: Detect form
	detection_result = detect_form_task(
		document_id=document_id,
		document_version_id=str(doc_ver.id),
		page_ids=page_ids,
		tenant_id=tenant_id,
	)

	# Step 2: Extract signatures if form detected
	signature_result = {"signature_count": 0, "signatures": []}
	if detection_result.get("is_form") or detection_result.get("signature_count", 0) > 0:
		signature_result = extract_signatures_task(
			document_id=document_id,
			document_version_id=str(doc_ver.id),
			page_ids=page_ids,
			tenant_id=tenant_id,
		)

	# Step 3: If template provided, extract specific fields
	if template_id:
		celery_app.send_task(
			"papermerge.core.tasks.extract_with_template",
			kwargs={
				"document_id": document_id,
				"template_id": template_id,
				"tenant_id": tenant_id,
			},
			route_name="default",
		)

	return {
		"document_id": document_id,
		"is_form": detection_result.get("is_form", False),
		"form_type": detection_result.get("form_type"),
		"field_count": detection_result.get("field_count", 0),
		"signature_count": signature_result.get("signature_count", 0),
	}


@shared_task(name="darchiva.signature.compare")
def compare_signatures_task(
	signature1_id: str,
	signature2_id: str,
	tenant_id: str,
) -> dict:
	"""Compare two signatures for similarity."""
	logger.info(f"Comparing signatures {signature1_id} and {signature2_id}")

	# Download signature images from S3
	sig1_data = s3.download_bytes(f"signatures/*/{signature1_id}.png")
	sig2_data = s3.download_bytes(f"signatures/*/{signature2_id}.png")

	if not sig1_data or not sig2_data:
		return {
			"error": "Could not load signature images",
			"similarity": 0.0,
		}

	extractor = SignatureExtractor()
	similarity = extractor.compare_signatures(sig1_data, sig2_data)

	return {
		"signature1_id": signature1_id,
		"signature2_id": signature2_id,
		"similarity": similarity,
		"is_match": similarity > 0.75,  # 75% threshold for match
	}
