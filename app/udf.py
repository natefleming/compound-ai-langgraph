from io import BytesIO
import warnings
import mimetypes
import json

from pyspark.sql import DataFrame

import pyspark.sql.functions as F
import pyspark.sql.types as T



from docling.document_converter import DocumentConverter, ConversionResult
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.document import DoclingDocument



pipeline_options: PdfPipelineOptions = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True


## Custom options are now defined per format.
def document_converter() -> DocumentConverter:
    converter: DocumentConverter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, # pipeline options go here.
                    backend=PyPdfiumDocumentBackend # optional: pick an alternative backend
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline # default for office formats and HTML
                ),
            },
        )
    )
    return converter

converter: DocumentConverter = document_converter()


def parse_bytes(raw_doc_contents_bytes: bytes, name: str) -> str:
    try:
      stream: BytesIO = BytesIO(raw_doc_contents_bytes)
      document_stream: DocumentStream = DocumentStream(name=name, stream=stream)
      result: ConversionResult = converter.convert(document_stream)
      document: DoclingDocument = result.document
      markdown: str = document.export_to_markdown()
      return markdown
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None


@F.udf(T.StringType())
def guess_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0]
  
  