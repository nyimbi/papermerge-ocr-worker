import logging
import tempfile
import time
from pathlib import Path
from uuid import UUID

import ocrmypdf
from pikepdf import Pdf

from .base import OCREngine, OCRResult, OCREngineType, TextLine, BoundingBox

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
	"""Tesseract OCR engine using ocrmypdf."""

	engine_type = OCREngineType.TESSERACT

	def __init__(self):
		self._languages: list[str] | None = None

	def ocr_image(
		self,
		image_path: Path,
		lang: str = 'eng',
		**kwargs
	) -> OCRResult:
		"""Perform OCR on a single image using Tesseract."""
		start_time = time.time()

		try:
			import pytesseract
			from PIL import Image

			image = Image.open(image_path)
			data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

			lines: list[TextLine] = []
			full_text_parts: list[str] = []
			confidences: list[float] = []

			current_line_text = []
			current_line_conf = []
			last_line_num = -1

			for i in range(len(data['text'])):
				text = data['text'][i].strip()
				conf = float(data['conf'][i])
				line_num = data['line_num'][i]

				if line_num != last_line_num and current_line_text:
					line_text = ' '.join(current_line_text)
					avg_conf = sum(current_line_conf) / len(current_line_conf) if current_line_conf else 0
					lines.append(TextLine(text=line_text, confidence=avg_conf))
					full_text_parts.append(line_text)
					current_line_text = []
					current_line_conf = []

				if text and conf > 0:
					current_line_text.append(text)
					current_line_conf.append(conf)
					confidences.append(conf)

				last_line_num = line_num

			if current_line_text:
				line_text = ' '.join(current_line_text)
				avg_conf = sum(current_line_conf) / len(current_line_conf) if current_line_conf else 0
				lines.append(TextLine(text=line_text, confidence=avg_conf))
				full_text_parts.append(line_text)

			full_text = '\n'.join(full_text_parts)
			avg_confidence = sum(confidences) / len(confidences) if confidences else 0

			processing_time = (time.time() - start_time) * 1000

			return OCRResult(
				text=full_text,
				confidence=avg_confidence / 100,
				lines=lines,
				language=lang,
				engine='tesseract',
				processing_time_ms=processing_time
			)

		except Exception as e:
			logger.error(f"Tesseract OCR failed for {image_path}: {e}")
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
		"""Perform OCR on a single PDF page using ocrmypdf."""
		start_time = time.time()

		preview_width = kwargs.get('preview_width', 300)
		sidecar_dir = kwargs.get('sidecar_dir', output_dir)

		pdf = Pdf.open(pdf_path)

		if page_number <= 0:
			raise ValueError("Page number must be at least 1")

		if page_number > len(pdf.pages):
			raise ValueError(
				f"File {pdf_path} has {len(pdf.pages)} pages. "
				f"Requested page {page_number} is out of range"
			)

		with Pdf.open(pdf_path) as pdf, tempfile.NamedTemporaryFile(suffix='.pdf') as temp:
			if len(pdf.pages) > 1:
				dst = Pdf.new()
				for n, page in enumerate(pdf.pages):
					if n + 1 == page_number:
						dst.pages.append(page)
						break
				dst.save(temp.name)
				dst.close()
				work_path = Path(temp.name)
			else:
				work_path = pdf_path

			ocrmypdf.ocr(
				work_path,
				output_dir,
				lang=lang,
				plugins=["ocrmypdf_papermerge.plugin"],
				progress_bar=False,
				output_type="pdf",
				pdf_renderer="hocr",
				use_threads=True,
				force_ocr=True,
				keep_temporary_files=False,
				sidecar_dir=sidecar_dir,
				uuid=str(page_uuid),
				pages="1",
				sidecar_format="svg",
				preview_width=preview_width,
				deskew=True,
			)

		sidecar_file = sidecar_dir / f"{page_uuid}.txt"
		text = ""
		if sidecar_file.exists():
			text = sidecar_file.read_text()

		processing_time = (time.time() - start_time) * 1000

		return OCRResult(
			text=text,
			confidence=0.9,
			language=lang,
			engine='tesseract',
			processing_time_ms=processing_time,
			page_number=page_number,
			metadata={'output_dir': str(output_dir), 'page_uuid': str(page_uuid)}
		)

	def is_available(self) -> bool:
		"""Check if Tesseract is installed."""
		try:
			import pytesseract
			pytesseract.get_tesseract_version()
			return True
		except Exception:
			return False

	def supported_languages(self) -> list[str]:
		"""Return list of installed Tesseract languages."""
		if self._languages is not None:
			return self._languages

		try:
			import pytesseract
			langs = pytesseract.get_languages()
			self._languages = [l for l in langs if l != 'osd']
			return self._languages
		except Exception:
			return ['eng']
