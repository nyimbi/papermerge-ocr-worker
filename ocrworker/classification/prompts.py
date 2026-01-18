"""Classification prompts for document type detection."""

CLASSIFICATION_PROMPTS = {
	'general': """Analyze this document image and classify it.

Determine:
1. DOCUMENT_TYPE: The specific type of document (invoice, receipt, contract, letter, form, report, legal_filing, medical_record, tax_form, bank_statement, insurance_claim, purchase_order, etc.)
2. CATEGORY: The general category (financial, legal, medical, correspondence, administrative, technical, personal, government)
3. CONFIDENCE: Your confidence level (high, medium, low)

Respond in exactly this format:
TYPE: <document_type>
CATEGORY: <category>
CONFIDENCE: <level>""",

	'financial': """Analyze this financial document and classify it.

Possible types:
- invoice: Bill for goods/services
- receipt: Proof of payment
- bank_statement: Account activity summary
- tax_form: Tax-related document (W-2, 1099, etc.)
- purchase_order: Order for goods/services
- credit_note: Credit memo/refund document
- expense_report: Employee expense documentation
- payslip: Salary/wage statement
- financial_statement: Balance sheet, income statement, etc.

Respond with ONLY the document type, nothing else.""",

	'legal': """Analyze this legal document and classify it.

Possible types:
- contract: Agreement between parties
- nda: Non-disclosure agreement
- power_of_attorney: Legal authorization
- deed: Property transfer document
- will: Testament
- court_filing: Legal court document
- affidavit: Sworn statement
- license: Permission or authorization
- terms_of_service: Service agreement terms
- privacy_policy: Data handling policy

Respond with ONLY the document type, nothing else.""",

	'medical': """Analyze this medical document and classify it.

Possible types:
- medical_record: Patient health information
- prescription: Medication order
- lab_result: Laboratory test results
- imaging_report: X-ray, MRI, CT results
- insurance_claim: Medical insurance claim
- referral: Referral to specialist
- discharge_summary: Hospital discharge document
- consent_form: Medical consent form
- vaccination_record: Immunization record

Respond with ONLY the document type, nothing else.""",

	'identity': """Analyze this identity document and classify it.

Possible types:
- passport: International travel document
- drivers_license: Driving permit
- national_id: National identity card
- birth_certificate: Birth record
- social_security: SSN card or document
- visa: Travel/work visa
- residence_permit: Residency document
- professional_license: Professional certification

Respond with ONLY the document type, nothing else.""",

	'correspondence': """Analyze this correspondence document and classify it.

Possible types:
- business_letter: Formal business communication
- personal_letter: Personal communication
- memo: Internal memorandum
- email_printout: Printed email
- fax: Facsimile document
- notice: Formal notification
- announcement: Public announcement

Respond with ONLY the document type, nothing else.""",

	'form': """Analyze this form document and identify its type.

Common form types:
- application_form: Job, loan, membership application
- registration_form: Registration for service/event
- survey_form: Questionnaire or survey
- order_form: Product/service order form
- feedback_form: Customer feedback form
- consent_form: Permission/consent form
- tax_form: Tax-related form
- insurance_form: Insurance application/claim

Respond with ONLY the form type, nothing else.""",

	'extract_metadata': """Extract key metadata from this document.

Identify and extract:
1. DOCUMENT_DATE: The document date (YYYY-MM-DD format)
2. DOCUMENT_NUMBER: Document/reference number
3. SENDER: Organization or person who sent/created this
4. RECIPIENT: Organization or person this is addressed to
5. AMOUNT: Total amount if financial document
6. CURRENCY: Currency if amount present

Respond in this format:
DATE: <date or NOT_FOUND>
NUMBER: <number or NOT_FOUND>
SENDER: <sender or NOT_FOUND>
RECIPIENT: <recipient or NOT_FOUND>
AMOUNT: <amount or NOT_FOUND>
CURRENCY: <currency or NOT_FOUND>""",

	'quality_check': """Assess the quality of this scanned document image.

Evaluate:
1. READABILITY: Can all text be clearly read? (good, fair, poor)
2. ALIGNMENT: Is the document properly aligned? (good, slight_skew, significant_skew)
3. COMPLETENESS: Is the entire document visible? (complete, partial, unclear)
4. CONTRAST: Is the contrast adequate? (good, fair, poor)
5. ISSUES: List any specific issues (blur, shadows, folds, stains, cut_off)

Respond in this format:
READABILITY: <level>
ALIGNMENT: <level>
COMPLETENESS: <level>
CONTRAST: <level>
ISSUES: <comma-separated list or NONE>""",
}
