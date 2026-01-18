# (c) Copyright Datacraft, 2026
"""Celery tasks for document segmentation."""
import logging
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

from celery import shared_task

from ocrworker import config
from ocrworker.db.engine import Session
from ocrworker.vlm.client import OllamaClient
from ocrworker.segmentation import (
	DocumentSegmenter,
	SegmentationMethod,
	SegmentationResult,
)

logger = logging.getLogger(__name__)
settings = config.get_settings()


@shared_task(name="ocrworker.segmentation_tasks.segment_document")
def segment_document(
	job_id: str,
	document_id: str,
	page_number: int | None = None,
	method: str = "hybrid",
	min_confidence: float = 0.6,
	deskew: bool = True,
	auto_create_documents: bool = False,
) -> dict:
	"""
	Segment a document scan to detect and extract multiple documents.

	This task:
	1. Downloads the document from S3
	2. Runs segmentation to detect document boundaries
	3. Extracts and saves segment images
	4. Creates ScanSegment records in the database
	5. Optionally creates new documents from segments

	Args:
		job_id: ID of the SegmentationJob record
		document_id: ID of document to segment
		page_number: Specific page to segment (None = all pages)
		method: Segmentation method (vlm, edge_detection, contour, hybrid)
		min_confidence: Minimum confidence threshold
		deskew: Whether to auto-deskew segments
		auto_create_documents: Whether to auto-create documents from segments

	Returns:
		Dict with job results
	"""
	start_time = time.time()
	logger.info(f"Starting segmentation job {job_id} for document {document_id}")

	result = {
		"job_id": job_id,
		"document_id": document_id,
		"status": "completed",
		"documents_detected": 0,
		"segments_created": 0,
		"error": None,
	}

	try:
		# Get document info from database
		with Session() as db_session:
			from ocrworker.db import get_last_version, get_doc_ver
			doc_ver = get_last_version(db_session, doc_id=UUID(document_id))

			if not doc_ver:
				raise ValueError(f"Document version not found for {document_id}")

		# Download document from S3
		from ocrworker import s3, plib
		s3.download_docver(doc_ver.id, doc_ver.file_name)
		doc_path = plib.abs_docver_path(doc_ver.id, doc_ver.file_name)

		# Initialize VLM client if using VLM or hybrid method
		vlm_client = None
		seg_method = SegmentationMethod(method)
		if seg_method in (SegmentationMethod.VLM, SegmentationMethod.HYBRID):
			vlm_client = OllamaClient(
				base_url=settings.ollama_base_url,
				model=settings.ollama_ocr_model,
				timeout=settings.ollama_timeout,
			)

		# Create output directory for segments
		output_dir = Path(settings.papermerge__main__media_root) / "segments" / job_id
		output_dir.mkdir(parents=True, exist_ok=True)

		# Initialize segmenter
		segmenter = DocumentSegmenter(
			vlm_client=vlm_client,
			method=seg_method,
			output_dir=output_dir,
			deskew=deskew,
			min_confidence=min_confidence,
		)

		# Convert PDF to images if needed
		import mimetypes
		_type, _ = mimetypes.guess_type(str(doc_path))

		if _type == "application/pdf":
			# Convert PDF pages to images for segmentation
			segments_results = _segment_pdf_pages(
				segmenter,
				doc_path,
				document_id,
				page_number,
			)
		else:
			# Single image
			import asyncio
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			try:
				seg_result = loop.run_until_complete(
					segmenter.segment(
						image_path=doc_path,
						original_scan_id=UUID(document_id),
					)
				)
				segments_results = [(1, seg_result)]
			finally:
				loop.close()

		# Process results and save to database
		total_segments = 0
		for page_num, seg_result in segments_results:
			total_segments += len(seg_result.segments)
			_save_segments_to_db(
				job_id=job_id,
				document_id=document_id,
				page_number=page_num,
				seg_result=seg_result,
			)

		result["documents_detected"] = sum(
			r.document_count for _, r in segments_results
		)
		result["segments_created"] = total_segments

		# Update job status
		_update_job_status(
			job_id=job_id,
			status="completed",
			documents_detected=result["documents_detected"],
			segments_created=result["segments_created"],
			processing_time_ms=(time.time() - start_time) * 1000,
		)

		logger.info(
			f"Segmentation job {job_id} completed: "
			f"{result['documents_detected']} documents, "
			f"{result['segments_created']} segments"
		)

	except Exception as e:
		logger.error(f"Segmentation job {job_id} failed: {e}")
		result["status"] = "failed"
		result["error"] = str(e)

		_update_job_status(
			job_id=job_id,
			status="failed",
			error_message=str(e),
			processing_time_ms=(time.time() - start_time) * 1000,
		)

	return result


def _segment_pdf_pages(
	segmenter: DocumentSegmenter,
	pdf_path: Path,
	document_id: str,
	page_number: int | None,
) -> list[tuple[int, SegmentationResult]]:
	"""Segment PDF pages."""
	import asyncio
	from pdf2image import convert_from_path

	# Convert PDF to images
	if page_number:
		images = convert_from_path(
			pdf_path,
			first_page=page_number,
			last_page=page_number,
			dpi=300,
		)
		page_numbers = [page_number]
	else:
		images = convert_from_path(pdf_path, dpi=300)
		page_numbers = list(range(1, len(images) + 1))

	results = []
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)

	try:
		for page_num, image in zip(page_numbers, images):
			# Save temp image
			temp_path = segmenter.output_dir / f"page_{page_num}_temp.png"
			image.save(temp_path, format='PNG')

			# Segment the page
			seg_result = loop.run_until_complete(
				segmenter.segment(
					image_path=temp_path,
					original_scan_id=UUID(document_id),
				)
			)

			results.append((page_num, seg_result))

			# Clean up temp file
			temp_path.unlink(missing_ok=True)

	finally:
		loop.close()

	return results


def _save_segments_to_db(
	job_id: str,
	document_id: str,
	page_number: int,
	seg_result: SegmentationResult,
) -> None:
	"""Save segment records to the database."""
	# Import here to avoid circular imports with papermerge-core
	# This task runs in the OCR worker which has access to the same DB
	from sqlalchemy import create_engine
	from sqlalchemy.orm import sessionmaker

	# Create direct DB session (OCR worker uses sync DB)
	engine = create_engine(settings.papermerge__database__url)
	SessionLocal = sessionmaker(bind=engine)

	with SessionLocal() as session:
		for segment in seg_result.segments:
			# Create segment record
			segment_data = {
				"original_scan_id": document_id,
				"original_page_number": page_number,
				"segment_number": segment.segment_number,
				"total_segments": segment.total_segments,
				"segmentation_confidence": segment.confidence,
				"segmentation_method": seg_result.method_used.value,
				"status": "pending",
				"was_deskewed": segment.was_deskewed,
				"rotation_angle": segment.deskew_angle,
				"document_type_hint": segment.document_type_hint,
				"segment_width": segment.width,
				"segment_height": segment.height,
				"segment_file_path": str(segment.image_path) if segment.image_path else None,
				"processing_time_ms": seg_result.processing_time_ms,
			}

			if segment.boundary:
				segment_data.update({
					"boundary_x": segment.boundary.x,
					"boundary_y": segment.boundary.y,
					"boundary_width": segment.boundary.width,
					"boundary_height": segment.boundary.height,
				})

			# Insert using raw SQL to avoid ORM dependency issues
			from sqlalchemy import text
			from uuid_extensions import uuid7str

			segment_id = uuid7str()
			segment_data["id"] = segment_id

			columns = ", ".join(segment_data.keys())
			placeholders = ", ".join(f":{k}" for k in segment_data.keys())

			session.execute(
				text(f"INSERT INTO scan_segments ({columns}) VALUES ({placeholders})"),
				segment_data,
			)

		session.commit()


def _update_job_status(
	job_id: str,
	status: str,
	documents_detected: int = 0,
	segments_created: int = 0,
	processing_time_ms: float = 0,
	error_message: str | None = None,
) -> None:
	"""Update segmentation job status in database."""
	from sqlalchemy import create_engine, text
	from sqlalchemy.orm import sessionmaker

	engine = create_engine(settings.papermerge__database__url)
	SessionLocal = sessionmaker(bind=engine)

	with SessionLocal() as session:
		completed_at = datetime.utcnow().isoformat() if status in ("completed", "failed") else None

		session.execute(
			text("""
				UPDATE segmentation_jobs
				SET status = :status,
					documents_detected = :documents_detected,
					segments_created = :segments_created,
					processing_time_ms = :processing_time_ms,
					error_message = :error_message,
					completed_at = :completed_at
				WHERE id = :job_id
			"""),
			{
				"job_id": job_id,
				"status": status,
				"documents_detected": documents_detected,
				"segments_created": segments_created,
				"processing_time_ms": processing_time_ms,
				"error_message": error_message,
				"completed_at": completed_at,
			},
		)
		session.commit()


@shared_task(name="ocrworker.segmentation_tasks.detect_multi_document")
def detect_multi_document(
	document_id: str,
	page_number: int = 1,
) -> dict:
	"""
	Quick check if a document page contains multiple documents.

	This is a lightweight task that only checks for multiple documents
	without actually extracting segments. Useful for auto-detection
	during upload.

	Args:
		document_id: ID of document to check
		page_number: Page to check

	Returns:
		Dict with detection results
	"""
	logger.info(f"Checking document {document_id} page {page_number} for multiple documents")

	try:
		# Get document
		with Session() as db_session:
			from ocrworker.db import get_last_version
			doc_ver = get_last_version(db_session, doc_id=UUID(document_id))

		if not doc_ver:
			return {"error": "Document not found", "is_multi_document": False}

		# Download and check
		from ocrworker import s3, plib
		s3.download_docver(doc_ver.id, doc_ver.file_name)
		doc_path = plib.abs_docver_path(doc_ver.id, doc_ver.file_name)

		# Initialize VLM client
		vlm_client = OllamaClient(
			base_url=settings.ollama_base_url,
			model=settings.ollama_ocr_model,
			timeout=30.0,  # Quick timeout for detection
		)

		segmenter = DocumentSegmenter(
			vlm_client=vlm_client,
			method=SegmentationMethod.VLM,
		)

		# Check for multi-document
		import mimetypes
		_type, _ = mimetypes.guess_type(str(doc_path))

		if _type == "application/pdf":
			from pdf2image import convert_from_path
			images = convert_from_path(
				doc_path,
				first_page=page_number,
				last_page=page_number,
				dpi=150,  # Lower DPI for quick check
			)
			if not images:
				return {"error": "Could not convert page", "is_multi_document": False}

			from PIL import Image
			import io
			img = images[0]
			temp_path = Path("/tmp") / f"multi_check_{document_id}_{page_number}.png"
			img.save(temp_path)

			try:
				import asyncio
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				result = loop.run_until_complete(
					segmenter._detect_multiple_documents(temp_path, img)
				)
			finally:
				loop.close()
				temp_path.unlink(missing_ok=True)

		else:
			from PIL import Image
			img = Image.open(doc_path)

			import asyncio
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			try:
				result = loop.run_until_complete(
					segmenter._detect_multiple_documents(doc_path, img)
				)
			finally:
				loop.close()

		return {
			"document_id": document_id,
			"page_number": page_number,
			"is_multi_document": result.get("is_multiple", False),
			"document_count": result.get("count", 1),
			"confidence": result.get("confidence", 0.5),
			"reason": result.get("reason", ""),
		}

	except Exception as e:
		logger.error(f"Multi-document detection failed: {e}")
		return {
			"error": str(e),
			"is_multi_document": False,
		}
