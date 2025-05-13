import re
import pdfplumber
from collections import Counter
from typing import List

# Defining regex patterns for detecting the start of the summary section and section headings
SUMMARY_START_PATTERN = re.compile(r"^(report\s+summary|summary|I. Summary)[:\s]*$", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(
    r"^(table of contents|contents|RAIU investigation|description of the occurrence|analysis|conclusions|measures taken|safety recommendations|additional information|list of abbreviations|glossary|references)",
    re.IGNORECASE
)

def extract_summary_section(pdf_path: str, header_detection_pages: int = 10) -> str:
    """
    Extracts the summary section from a PDF document.
    It tries to identify and remove repeating headers first.
    If no summary section is found, it returns the full text.

    Args:
        pdf_path (str): The path to the PDF file.
        header_detection_pages (int): Number of initial pages to scan for repeating headers.

    Returns:
        str: The extracted summary text or full text if summary is not found.
    """
    summary_text = ""
    full_text_capture = ""
    capturing = False
    detected_header = None
    first_lines_for_header_detection: List[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            pages_to_check_for_header = min(header_detection_pages, num_pages)

            # Step 1: Detect a repeating header line across the first few pages
            for i in range(pages_to_check_for_header):
                page_text = pdf.pages[i].extract_text()
                if not page_text:
                    continue
                lines = page_text.split("\n")
                if lines:
                    first_lines_for_header_detection.append(lines[0].strip())

            # Step 2: Determine most common first line (if repeated)
            if first_lines_for_header_detection:
                first_line_counts = Counter(first_lines_for_header_detection)
                if first_line_counts: # Ensure there's at least one line
                    most_common_line, count = first_line_counts.most_common(1)[0]
                    # Header appears on a significant percentage of sampled pages
                    if count >= pages_to_check_for_header * 0.7 and count > 1: # Ensure it's actually repeating
                        detected_header = most_common_line
                        print(f"[INFO] Detected consistent header: '{detected_header}'")

            # Step 3: Process all pages and remove header if matched
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if not page_text:
                    continue

                page_lines = page_text.split("\n")
                current_page_content_lines = []

                if detected_header and page_lines and page_lines[0].strip() == detected_header:
                    current_page_content_lines = page_lines[1:]  # Remove header
                else:
                    current_page_content_lines = page_lines

                # Append to full text capture
                full_text_capture += f"[Page {page.page_number}]\n" + "\n".join(current_page_content_lines) + "\n\n"

                # Logic for capturing summary
                for line_num, line in enumerate(current_page_content_lines):
                    stripped_line = line.strip()
                    if not capturing and SUMMARY_START_PATTERN.match(stripped_line):
                        print(f"[INFO] Found summary start on Page {page.page_number}: '{stripped_line}'")
                        capturing = True
                        # Potentially skip the heading line itself from being added to summary_text
                        if SUMMARY_START_PATTERN.match(stripped_line).group(0) == stripped_line:
                            if ":" in stripped_line:
                                summary_text += stripped_line.split(":", 1)[1].strip() + "\n"
                            continue # Skip the heading line itself if it was just the heading
                        else: # Content on the same line as the heading
                            summary_text += stripped_line[len(SUMMARY_START_PATTERN.match(stripped_line).group(0)):].strip() + "\n"
                            continue


                    if capturing and SECTION_HEADING_PATTERN.match(stripped_line):
                        # Check if this heading is very close to the start of a page,
                        # it might be a false positive for ending the summary if the summary is short.
                        is_false_positive_section_heading = (line_num < 2 and len(summary_text.split()) < 50)

                        if not is_false_positive_section_heading:
                            print(f"[INFO] Stopping capture at heading on Page {page.page_number}: '{stripped_line}'")
                            capturing = False
                            break # Stop processing lines on this page for the summary
                        else:
                            print(f"[INFO] Ignored potential section heading '{stripped_line}' on Page {page.page_number} as likely part of summary.")


                    if capturing:
                        summary_text += line + "\n"
                
                if not capturing and summary_text: # If summary capture ended on this page
                    break # Stop processing further pages for summary

    except Exception as e:
        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
        return full_text_capture.strip()

    if not summary_text:
        print(f"[INFO] No specific summary section found in {pdf_path} using patterns. Returning full text.")
        return full_text_capture.strip()

    return summary_text.strip()
