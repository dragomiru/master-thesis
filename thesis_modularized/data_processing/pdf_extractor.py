# --- General modules ---
import re
import pdfplumber
from collections import Counter
from typing import List

# --- Patterns for summary identification ---
SUMMARY_START_PATTERN = re.compile(r"^(report\s+summary|summary|I\.\s+Summary)[:\s]*$", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(
    r"^(table of contents|introduction|contents|RAIU investigation|description of the occurrence|analysis|conclusions|measures taken|safety recommendations|additional information|list of abbreviations|glossary|references|II\. THE INVESTIGATION AND ITS CONTEXT|II\. INVESTIGATION AND ITS CONTEXT)",
    re.IGNORECASE
)

# --- Function for extracting the text from PDF ---
def get_pdf_text(
    pdf_path: str,
    summary_only: bool = True,
    header_detection_pages: int = 10
) -> str:
    """
    Extracts text from a PDF. If summary_only is True, it attempts to find and return
    only the summary section, falling back to full text if no summary is found.
    If summary_only is False, it returns the full text of the document.
    """
    summary_text = ""
    full_text_capture = ""
    capturing_summary = False
    detected_header = None
    first_lines = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            pages_to_check = min(header_detection_pages, num_pages)

            # Step 1 & 2: Detect and determine repeating header
            if pages_to_check > 0:
                for i in range(pages_to_check):
                    page_text = pdf.pages[i].extract_text()
                    if page_text and page_text.split("\n")[0].strip():
                        first_lines.append(page_text.split("\n")[0].strip())
            
            if first_lines:
                counts = Counter(first_lines)
                if counts:
                    most_common, num_most_common = counts.most_common(1)[0]
                    if num_most_common > 1 and num_most_common >= pages_to_check * 0.7:
                        detected_header = most_common
                        print(f"[INFO] Detected consistent header: '{detected_header}'")

            # Step 3: Process all pages
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue

                lines = page_text.split("\n")
                if detected_header and lines and lines[0].strip() == detected_header:
                    lines = lines[1:]

                # Always capture the full text (post-header removal)
                full_text_capture += "\n".join(lines) + "\n"

                # If we only want the summary, perform the summary capture logic
                if summary_only:
                    for line in lines:
                        stripped = line.strip()
                        if not capturing_summary and SUMMARY_START_PATTERN.match(stripped):
                            capturing_summary = True
                            continue

                        if capturing_summary and SECTION_HEADING_PATTERN.match(stripped):
                            capturing_summary = False
                            break # Stop processing this page's lines

                        if capturing_summary:
                            summary_text += line + "\n"
                    
                    # If we found a summary and are no longer capturing, we can stop processing pages
                    if not capturing_summary and summary_text:
                        break

    except Exception as e:
        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
        return full_text_capture.strip()

    # --- Return logic ---
    if summary_only:
        if summary_text:
            print("[INFO] Returning extracted summary text.")
            return summary_text.strip()
        else:
            print("[INFO] No summary found. Falling back to full text.")
            return full_text_capture.strip()
    else: # If summary_only was False from the start
        print("[INFO] Returning full document text as requested.")
        return full_text_capture.strip()
