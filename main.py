"""
Telegram Mini App Backend - Passport MRZ Scanner using Mistral AI Vision API
Production-ready FastAPI application with intelligent MRZ extraction
"""

import io
import os
import time
import base64
import hashlib
import json
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from mistralai import Mistral
from PIL import Image
from telegram import Update

app = FastAPI(
    title="Passport Scanner with Mistral AI",
    description="Telegram Mini App for Passport MRZ Scanning",
    version="3.0.0"
)

# CORS middleware for Telegram Mini App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# ============================================
# MISTRAL AI VISION SCANNER
# ============================================

class MistralMRZScanner:
    """
    Mistral AI OCR API Manager for MRZ Extraction
    Uses Mistral OCR API for passport scanning
    """

    def __init__(self, api_key: str):
        # Initialize with 60-second timeout to prevent hanging
        self.client = Mistral(
            api_key=api_key,
            timeout_ms=60000  # 60 second timeout for API calls
        )
        self.ocr_model = "mistral-ocr-latest"
        self.extraction_model = "mistral-small-latest"  # Lighter model for text extraction
        print(f"🤖 Mistral MRZ Scanner initialized", flush=True)
        print(f"   OCR Model: {self.ocr_model}", flush=True)
        print(f"   Extraction Model: {self.extraction_model}", flush=True)
        print(f"⏱️  Timeout configured: 60 seconds", flush=True)

    def scan_passport_mrz(self, image_bytes: bytes, max_retries: int = 3) -> Dict:
        """
        Scan passport MRZ using Mistral OCR API
        Returns extracted MRZ data with retry logic
        """
        print(f"🔍 Starting MRZ scan with Mistral OCR API...", flush=True)
        print(f"📊 Image size: {len(image_bytes)} bytes", flush=True)
        print(f"🔄 Max retries: {max_retries}", flush=True)

        attempts = 0
        last_error = None

        while attempts < max_retries:
            try:
                # Convert image to base64
                base64_image = base64.b64encode(image_bytes).decode('utf-8')

                print(f"🤖 Sending request to Mistral OCR API (attempt {attempts + 1})...", flush=True)

                # Use Mistral OCR API as per official documentation
                ocr_response = self.client.ocr.process(
                    model=self.ocr_model,
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    include_image_base64=True
                )

                if not ocr_response:
                    raise Exception("Empty response from Mistral OCR API")

                print(f"✅ Received response from Mistral OCR API", flush=True)

                # Extract text from OCR response
                # The OCR API returns structured data, we need to extract the MRZ text
                response_text = self._extract_ocr_text(ocr_response)
                print(f"📝 Extracted OCR text: {response_text[:100]}...", flush=True)

                # Try to parse as JSON first
                try:
                    result = self._parse_as_json(response_text)
                    print(f"✅ Successfully parsed as JSON", flush=True)
                except Exception as json_error:
                    # If JSON parsing fails, use Mistral to extract structured data
                    print(f"⚠️  JSON parsing failed, using Mistral extraction model...", flush=True)
                    result = self._extract_mrz_with_mistral(response_text)
                    print(f"✅ Extracted MRZ using Mistral model", flush=True)

                if "error" in result:
                    raise Exception(result["error"])

                # Validate MRZ format
                if not self._validate_mrz_format(result):
                    raise Exception("Invalid MRZ format from Mistral OCR API")

                print(f"✅ MRZ extracted successfully", flush=True)
                return result

            except Exception as e:
                error_msg = str(e).lower()
                error_type = type(e).__name__
                last_error = e
                attempts += 1

                print(f"❌ Attempt {attempts} failed ({error_type}): {str(e)[:150]}", flush=True)

                # Check if error is timeout-related
                if "timeout" in error_msg or "timed out" in error_msg:
                    print(f"⏱️  Timeout error - Mistral API did not respond in time", flush=True)
                    if attempts < max_retries:
                        print(f"⚠️ Retrying with exponential backoff...", flush=True)
                        time.sleep(2 ** attempts)  # Exponential backoff: 2s, 4s, 8s
                    continue
                # Check if error is related to rate limit
                elif "429" in error_msg or "rate limit" in error_msg:
                    print(f"⚠️ Rate limit error detected, waiting before retry...", flush=True)
                    time.sleep(3)
                    continue
                # Check if error is authentication/API key issue
                elif "401" in error_msg or "unauthorized" in error_msg or "api key" in error_msg:
                    print(f"🔑 Authentication error - Invalid or expired Mistral API key", flush=True)
                    raise HTTPException(
                        status_code=500,
                        detail="Mistral API authentication failed. Please check API key configuration."
                    )
                elif "503" in error_msg or "500" in error_msg:
                    print(f"⚠️ Server error, waiting before retry...", flush=True)
                    time.sleep(1)
                    continue
                else:
                    # Other errors - still retry
                    if attempts < max_retries:
                        print(f"⚠️ Unknown error, retrying...", flush=True)
                        time.sleep(0.5)
                    continue

        # All retries failed
        error_type = type(last_error).__name__
        error_detail = str(last_error)[:200]

        print(f"❌ All {max_retries} attempts failed. Last error type: {error_type}", flush=True)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan passport after {max_retries} attempts. Error: {error_type} - {error_detail}"
        )

    def _extract_ocr_text(self, ocr_response) -> str:
        """Extract text from Mistral OCR API response"""
        import html
        import re

        try:
            # Mistral OCR API returns a response with .pages attribute
            # Each page has a .markdown attribute with the extracted text
            if hasattr(ocr_response, 'pages'):
                # Extract markdown from all pages
                all_text = []
                for page in ocr_response.pages:
                    if hasattr(page, 'markdown'):
                        all_text.append(page.markdown)

                # Join all text
                full_text = '\n'.join(all_text)

                # Decode HTML entities (e.g., &lt; -> <, &gt; -> >)
                full_text = html.unescape(full_text)

                # OCR Error Corrections for MRZ-specific patterns
                # Fix 1: Remove HTML/XML-like tags that OCR might create
                full_text = re.sub(r'</[^>]+>', '', full_text)  # Remove closing tags like </ugli>

                # Fix 2: Convert ">< " pattern to "<<" (common OCR error)
                full_text = full_text.replace('><', '<<')

                # Fix 3: Fix spacing issues around chevrons
                full_text = full_text.replace('< ', '<').replace(' <', '<')

                print(f"📝 Extracted markdown text: {full_text[:200]}", flush=True)

                # Find MRZ lines in the text
                mrz_lines = self._extract_mrz_lines(full_text)
                if len(mrz_lines) >= 2:
                    return json.dumps({"line1": mrz_lines[0], "line2": mrz_lines[1]})

                return full_text

            # Fallback: try other common attributes
            elif hasattr(ocr_response, 'text'):
                return html.unescape(ocr_response.text)
            elif hasattr(ocr_response, 'content'):
                return html.unescape(ocr_response.content)
            elif isinstance(ocr_response, dict):
                # If it's a dictionary, try common keys
                for key in ['text', 'content', 'ocr_text', 'result']:
                    if key in ocr_response:
                        return html.unescape(str(ocr_response[key]))

            # If we can't find the text, convert to string and try to parse
            response_str = str(ocr_response)
            response_str = html.unescape(response_str)
            print(f"⚠️ OCR response format: {response_str[:200]}", flush=True)

            # Try to find MRZ lines
            mrz_lines = self._extract_mrz_lines(response_str)
            if len(mrz_lines) >= 2:
                return json.dumps({"line1": mrz_lines[0], "line2": mrz_lines[1]})

            # If still nothing found, return the full response for parsing
            return response_str

        except Exception as e:
            print(f"❌ Error extracting OCR text: {str(e)}", flush=True)
            return html.unescape(str(ocr_response))

    def _smart_split_mrz(self, text: str) -> tuple:
        """
        Intelligently split concatenated MRZ by finding passport number pattern
        Uzbek passports: 2 letters + 7 digits (e.g., FA1234567, FB0292047)
        """
        import re

        # Pattern: 2 uppercase letters followed by 7 digits
        # This is where Line 2 starts (passport number)
        passport_pattern = re.compile(r'[A-Z]{2}\d{7}', re.IGNORECASE)

        match = passport_pattern.search(text)
        if match:
            split_pos = match.start()
            print(f"🎯 Found passport number at position {split_pos}: {match.group()}", flush=True)

            line1 = text[:split_pos]
            line2 = text[split_pos:]

            # Normalize both lines to exactly 44 chars
            line1 = self._normalize_mrz_line(line1)
            line2 = self._normalize_mrz_line(line2[:44])  # Take only first 44 chars of line2

            return (line1, line2)

        # Fallback: blind split at 44
        print(f"⚠️  Passport pattern not found, using position 44", flush=True)
        return (text[:44], text[44:88] if len(text) >= 88 else text[44:])

    def _extract_mrz_lines(self, text: str) -> list:
        """Extract MRZ lines from text"""
        # MRZ lines for TD3 (passport) format:
        # - Line 1: 44 characters, starts with 'P<'
        # - Line 2: 44 characters, contains passport number, dates, etc.
        # Both lines use '<' as filler character

        lines = text.split('\n')
        mrz_lines = []

        for line in lines:
            # Clean the line
            line = line.strip()

            # Check for concatenated MRZ (80+ characters)
            if len(line) >= 80 and (line.upper().startswith('P<') or 'P<' in line[:5].upper()):
                # Use smart split to find passport number
                line1, line2 = self._smart_split_mrz(line)
                mrz_lines.append(line1)
                mrz_lines.append(line2)
                print(f"🔀 Smart split MRZ: {len(line)} chars → 2 lines", flush=True)
                continue

            # MRZ lines are exactly 44 characters and contain multiple '<' characters
            if len(line) == 44:
                # Check if line contains typical MRZ patterns
                if line.count('<') >= 3:  # MRZ lines have many '<' characters
                    mrz_lines.append(line)
            # Also check for lines that start with P< (first MRZ line)
            elif line.upper().startswith('P<') and len(line) >= 40:
                # Pad or trim to exactly 44 characters
                if len(line) < 44:
                    line = line.ljust(44, '<')
                else:
                    line = line[:44]
                mrz_lines.append(line)

        print(f"🔍 Found {len(mrz_lines)} potential MRZ lines", flush=True)
        for i, line in enumerate(mrz_lines):
            print(f"   Line {i+1}: {line} (length: {len(line)})", flush=True)

        return mrz_lines

    def _parse_as_json(self, response_text: str) -> Dict:
        """Parse response text as JSON"""
        import json
        import re

        # Clean response (remove markdown code blocks if present)
        cleaned = response_text.strip()

        # Remove markdown code blocks
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}", flush=True)
            print(f"Response text: {response_text[:200]}", flush=True)
            raise Exception("Invalid JSON response from Mistral OCR API")

    def _extract_mrz_with_mistral(self, ocr_text: str) -> Dict:
        """
        Use Mistral chat model to extract structured MRZ data from OCR text
        This is used when OCR returns plain text instead of structured JSON
        """
        import json
        import re
        import html

        # Decode HTML entities first
        ocr_text = html.unescape(ocr_text)

        # Filter out garbage lines (like all zeros)
        lines = ocr_text.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are all the same character repeated
            if line and not all(c == line[0] for c in line):
                # Skip lines that don't contain typical MRZ characters
                if 'P<' in line or '<' in line or any(c.isalpha() for c in line):
                    filtered_lines.append(line)

        cleaned_ocr_text = '\n'.join(filtered_lines)
        print(f"🧹 Cleaned OCR text: {cleaned_ocr_text[:200]}", flush=True)

        # Try manual extraction first (faster and more reliable)
        manual_result = self._manual_mrz_extraction(cleaned_ocr_text)
        if manual_result:
            return manual_result

        # Fallback to Mistral if manual extraction fails
        prompt = f"""Extract passport MRZ lines from this OCR text. Return ONLY valid JSON with no markdown.

OCR Text:
{cleaned_ocr_text}

Requirements:
- Extract TWO lines, each EXACTLY 44 characters
- Line 1 starts with 'P<' (document type + country + name)
- Line 2 has passport number, nationality, dates
- Use '<' as filler
- If concatenated, split at position 44
- Pad short lines with '<' at the end
- Truncate long lines to 44 characters

Return this exact JSON format (no markdown):
{{"line1": "44-char line 1", "line2": "44-char line 2"}}"""

        print(f"🔄 Sending to Mistral extraction model...", flush=True)

        # Call Mistral chat API without response_format (not supported)
        response = self.client.chat.complete(
            model=self.extraction_model,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract content from response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            print(f"📝 Mistral response: {content[:300]}", flush=True)

            if not content or content.strip() == "":
                raise Exception("Empty response from Mistral extraction model")

            # Aggressively clean the response
            cleaned = content.strip()
            # Remove ALL markdown
            cleaned = re.sub(r'```[a-z]*\s*', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
            cleaned = cleaned.strip()

            # Try to find JSON object in the response
            json_match = re.search(r'\{[^}]*"line1"[^}]*"line2"[^}]*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            if not cleaned:
                raise Exception("No valid JSON found in Mistral response")

            try:
                result = json.loads(cleaned)

                if "line1" not in result or "line2" not in result:
                    raise Exception("Missing line1 or line2")

                # Ensure exactly 44 characters
                result["line1"] = self._normalize_mrz_line(result["line1"])
                result["line2"] = self._normalize_mrz_line(result["line2"])

                print(f"   Line1 ({len(result['line1'])}): {result['line1']}", flush=True)
                print(f"   Line2 ({len(result['line2'])}): {result['line2']}", flush=True)

                return result
            except (json.JSONDecodeError, Exception) as e:
                print(f"❌ Mistral extraction failed: {e}", flush=True)
                raise Exception(f"Failed to extract MRZ: {str(e)}")
        else:
            raise Exception("Empty response from Mistral extraction model")

    def _manual_mrz_extraction(self, text: str) -> Optional[Dict]:
        """Manually extract MRZ lines from text without AI"""
        import html
        import re

        # Decode HTML entities
        text = html.unescape(text)

        # Apply OCR error corrections
        text = re.sub(r'</[^>]+>', '', text)  # Remove closing tags
        text = text.replace('><', '<<')  # Fix >< to <<
        text = text.replace('< ', '<').replace(' <', '<')  # Remove spaces around <

        lines = text.split('\n')
        mrz_candidates = []

        for line in lines:
            line = line.strip()
            # Apply same corrections to each line
            line = re.sub(r'</[^>]+>', '', line)
            line = line.replace('><', '<<')
            line = line.replace('< ', '<').replace(' <', '<')

            # Look for lines that start with P< or have MRZ characteristics
            if line.startswith('P<') or line.startswith('p<') or (len(line) >= 40 and '<' in line):
                mrz_candidates.append(line)

        # Try concatenated format first (most common issue)
        for line in mrz_candidates:
            if len(line) >= 80:  # Two lines concatenated (at least 80 chars)
                # Use smart split to find passport number pattern
                line1, line2 = self._smart_split_mrz(line)

                print(f"✅ Manual extraction (concatenated {len(line)} chars) succeeded", flush=True)
                print(f"   Line1 ({len(line1)}): {line1}", flush=True)
                print(f"   Line2 ({len(line2)}): {line2}", flush=True)

                return {"line1": line1, "line2": line2}

        # Try separate lines
        if len(mrz_candidates) >= 2:
            line1 = self._normalize_mrz_line(mrz_candidates[0])
            line2 = self._normalize_mrz_line(mrz_candidates[1])

            print(f"✅ Manual extraction (separate lines) succeeded", flush=True)
            print(f"   Line1 ({len(line1)}): {line1}", flush=True)
            print(f"   Line2 ({len(line2)}): {line2}", flush=True)

            return {"line1": line1, "line2": line2}

        # If we have only 1 candidate, check if it's at least close to concatenated
        if len(mrz_candidates) == 1:
            line = mrz_candidates[0]
            if len(line) >= 70:  # Might be concatenated but with some chars missing
                # Use smart split for partial concatenation too
                line1, line2 = self._smart_split_mrz(line)

                print(f"✅ Manual extraction (partial concatenated {len(line)} chars) succeeded", flush=True)
                print(f"   Line1 ({len(line1)}): {line1}", flush=True)
                print(f"   Line2 ({len(line2)}): {line2}", flush=True)

                return {"line1": line1, "line2": line2}

        print(f"⚠️  Manual extraction found {len(mrz_candidates)} candidates, need 2", flush=True)
        if mrz_candidates:
            for i, cand in enumerate(mrz_candidates):
                print(f"   Candidate {i+1}: {len(cand)} chars - {cand[:50]}...", flush=True)
        return None

    def _normalize_mrz_line(self, line: str) -> str:
        """Normalize MRZ line to exactly 44 characters"""
        import html
        import re

        # Decode HTML entities
        line = html.unescape(line)

        # Apply OCR error corrections
        line = re.sub(r'</[^>]+>', '', line)  # Remove closing tags like </name>
        line = line.replace('><', '<<')  # Fix >< to <<
        line = line.replace('< ', '<').replace(' <', '<')  # Remove spaces around <

        # Remove any other whitespace
        line = line.strip()

        # Convert to uppercase (MRZ standard)
        line = line.upper()

        # Truncate or pad to exactly 44 characters
        if len(line) > 44:
            line = line[:44]
        elif len(line) < 44:
            line = line.ljust(44, '<')

        return line

    def _validate_mrz_format(self, result: Dict) -> bool:
        """Validate MRZ format (2 lines, 44 chars each)"""
        if "line1" not in result or "line2" not in result:
            print(f"❌ Missing line1 or line2 in response", flush=True)
            return False

        line1 = result["line1"]
        line2 = result["line2"]

        if len(line1) != 44:
            print(f"❌ Line1 length is {len(line1)}, expected 44", flush=True)
            return False

        if len(line2) != 44:
            print(f"❌ Line2 length is {len(line2)}, expected 44", flush=True)
            return False

        return True


# ============================================
# ICAO 9303 MRZ PARSER
# ============================================

class MRZParser:
    """Parse and validate ICAO 9303 TD3 format MRZ"""

    @staticmethod
    def char_to_value(char: str) -> int:
        """Convert MRZ character to numeric value for checksum"""
        if char.isdigit():
            return int(char)
        elif char.isalpha():
            return ord(char) - ord('A') + 10
        elif char == '<':
            return 0
        else:
            return 0

    @staticmethod
    def calculate_checksum(data: str) -> int:
        """Calculate ICAO 9303 checksum using mod 10 with weights 7,3,1"""
        weights = [7, 3, 1]
        total = 0

        for i, char in enumerate(data):
            value = MRZParser.char_to_value(char)
            weight = weights[i % 3]
            total += value * weight

        return total % 10

    @staticmethod
    def validate_checksum(data: str, check_digit: str) -> bool:
        """Validate data against its check digit"""
        if not check_digit.isdigit():
            return False

        calculated = MRZParser.calculate_checksum(data)
        expected = int(check_digit)

        return calculated == expected

    @staticmethod
    def format_date(yymmdd: str) -> str:
        """Convert YYMMDD to DD.MM.YYYY format"""
        if len(yymmdd) != 6 or not yymmdd.isdigit():
            return yymmdd

        yy = int(yymmdd[0:2])
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]

        # Determine century (20xx for years < 50, 19xx otherwise)
        yyyy = 2000 + yy if yy < 50 else 1900 + yy

        return f"{dd}.{mm}.{yyyy}"

    @staticmethod
    def ocr_error_correction_numbers(text: str) -> str:
        """Correct common OCR errors in number fields (O -> 0)"""
        return text.replace('O', '0').replace('o', '0')

    @staticmethod
    def ocr_error_correction_letters(text: str) -> str:
        """Correct common OCR errors in letter fields (0 -> O)"""
        return text.replace('0', 'O')

    @staticmethod
    def parse_mrz(line1: str, line2: str) -> Dict:
        """
        Parse complete TD3 MRZ (2 lines x 44 chars) following ICAO 9303 standard STRICTLY
        Line 1: P<CCCSURNAME<<GIVEN<NAMES<<<<<<<<<<<<<<<<<<
        Line 2: NNNNNNNNNNCYYYMMDDCSXYYYMMDDCZZZZZZZZZZZZZCC

        CRITICAL: Uses FIXED character positions for Line 2 parsing
        """
        # Clean and uppercase
        line1 = line1.strip().upper()
        line2 = line2.strip().upper()

        print(f"🔍 Parsing MRZ lines:", flush=True)
        print(f"   Line 1: {line1}", flush=True)
        print(f"   Line 2: {line2}", flush=True)

        if len(line1) != 44 or len(line2) != 44:
            raise ValueError(f"Invalid MRZ format. Line1: {len(line1)}, Line2: {len(line2)}")

        # ===== LINE 1 PARSING (Name Parsing with << separator) =====
        doc_type = line1[0]  # Should be 'P' for passport
        country_code = line1[2:5].replace('<', '').strip()

        # Name section starts at position 5
        names_section = line1[5:44]

        # Find the double chevron separator <<
        if '<<' in names_section:
            separator_pos = names_section.index('<<')
            surname_raw = names_section[:separator_pos]
            given_names_raw = names_section[separator_pos + 2:]  # Skip <<

            # Clean surname: replace single < with space, remove trailing <
            surname = surname_raw.replace('<', ' ').strip()

            # Clean given names: replace single < with space, remove trailing <
            given_names = given_names_raw.replace('<', ' ').strip()
        else:
            # Fallback: no clear separator
            surname = names_section.replace('<', ' ').strip()
            given_names = ""

        # Apply OCR correction to names (0 -> O)
        surname = MRZParser.ocr_error_correction_letters(surname)
        given_names = MRZParser.ocr_error_correction_letters(given_names)

        # ===== LINE 2 PARSING (STRICT FIXED POSITIONS) =====
        # Extract by exact indices as per ICAO 9303
        passport_number_raw = line2[0:9]
        passport_check = line2[9]
        nationality = line2[10:13]
        dob_raw = line2[13:19]
        dob_check = line2[19]
        sex = line2[20]
        expiry_raw = line2[21:27]
        expiry_check = line2[27]
        personal_number_raw = line2[28:42]  # 14 characters (PINFL for UZB)
        personal_check = line2[42]
        composite_check = line2[43]

        # Apply OCR corrections
        # For passport number: first 2 chars should be LETTERS, rest should be DIGITS
        passport_cleaned = passport_number_raw.replace('<', '').strip()
        if len(passport_cleaned) >= 2:
            # Ensure first 2 characters are letters (fix O->0 errors)
            prefix = passport_cleaned[:2].replace('0', 'O').replace('1', 'I')
            # Ensure remaining characters are digits (fix O->0 errors)
            suffix = passport_cleaned[2:].replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
            passport_number = prefix + suffix
        else:
            passport_number = MRZParser.ocr_error_correction_numbers(passport_cleaned)

        # Fix common nationality OCR errors (especially for UZB)
        nationality_raw = nationality.replace('<', '').strip()
        if nationality_raw in ['ZBO', 'LZB', 'USB', 'U2B', 'UZ8', 'UZD', '028', 'O2B']:
            nationality = 'UZB'
        else:
            nationality = nationality_raw

        dob = MRZParser.ocr_error_correction_numbers(dob_raw)
        expiry = MRZParser.ocr_error_correction_numbers(expiry_raw)
        personal_number = MRZParser.ocr_error_correction_numbers(personal_number_raw).replace('<', '').strip()
        sex = sex.replace('<', '')

        # Validate checksums
        validations = {
            "passport_number_valid": MRZParser.validate_checksum(passport_number_raw, passport_check),
            "dob_valid": MRZParser.validate_checksum(dob_raw, dob_check),
            "expiry_valid": MRZParser.validate_checksum(expiry_raw, expiry_check),
            "personal_number_valid": MRZParser.validate_checksum(personal_number_raw, personal_check),
        }

        # Composite check (entire line 2 except last check digit)
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        validations["composite_valid"] = MRZParser.validate_checksum(composite_data, composite_check)

        # Validate date formats
        validations["dob_format_valid"] = len(dob) == 6 and dob.isdigit()
        validations["expiry_format_valid"] = len(expiry) == 6 and expiry.isdigit()

        # Overall validation
        all_checks_valid = all(validations.values())

        print(f"✅ Parsed data:", flush=True)
        print(f"   Name: {given_names} {surname}", flush=True)
        print(f"   Passport: {passport_number}", flush=True)
        print(f"   PINFL: {personal_number}", flush=True)
        print(f"   Validation: {'PASS' if all_checks_valid else 'FAIL'}", flush=True)

        return {
            "surname": surname,
            "name": given_names,
            "passport_number": passport_number,
            "birth_date": MRZParser.format_date(dob),
            "expiry_date": MRZParser.format_date(expiry),
            "sex": sex if sex in ['M', 'F'] else 'F',  # Default to F if unclear
            "nationality": nationality,
            "pinfl": personal_number,  # 14-digit PINFL for Uzbek passports
            # Legacy fields for backwards compatibility
            "document_type": doc_type,
            "country_code": country_code,
            "given_names": given_names,
            "date_of_birth": MRZParser.format_date(dob),
            "date_of_birth_raw": dob,
            "date_of_expiry": MRZParser.format_date(expiry),
            "date_of_expiry_raw": expiry,
            "personal_number": personal_number,
            "validations": validations,
            "validation_status": "PASS" if all_checks_valid else "FAIL",
            "raw_mrz": {
                "line1": line1,
                "line2": line2
            }
        }


# ============================================
# SECURITY & RATE LIMITING
# ============================================

request_timestamps = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 20
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def check_rate_limit(client_id: str) -> bool:
    """Rate limiting: Max 20 requests per 60 seconds per client"""
    current_time = time.time()

    if client_id not in request_timestamps:
        request_timestamps[client_id] = []

    # Remove old timestamps
    request_timestamps[client_id] = [
        ts for ts in request_timestamps[client_id]
        if current_time - ts < RATE_LIMIT_WINDOW
    ]

    # Check limit
    if len(request_timestamps[client_id]) >= MAX_REQUESTS_PER_WINDOW:
        return False

    request_timestamps[client_id].append(current_time)
    return True

def validate_image_file(file: UploadFile, contents: bytes) -> None:
    """Validate uploaded file for security"""
    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
        )

    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    # Validate it's actually an image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )


# ============================================
# API ENDPOINTS
# ============================================

# Initialize Mistral Scanner
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "JWSVnIJhbnyhc80PY32AhKkxEbS4SFFi")

# Log API key configuration
if os.environ.get("MISTRAL_API_KEY"):
    print(f"✅ Using MISTRAL_API_KEY from environment variable", flush=True)
else:
    print(f"⚠️  WARNING: Using hardcoded Mistral API key (set MISTRAL_API_KEY env var)", flush=True)

mistral_scanner = MistralMRZScanner(api_key=MISTRAL_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the Telegram Mini App frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "passport-scanner-mistral",
        "version": "3.0.0"
    }


@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint for passport scanning using Mistral AI Vision API
    """
    try:
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"

        if not check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment."
            )

        # Read file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Validate file
        validate_image_file(file, contents)

        print(f"📸 Processing image: {file.filename} ({len(contents)} bytes)", flush=True)

        # Scan with Mistral AI
        print("🤖 Scanning passport with Mistral OCR API...", flush=True)
        mrz_result = mistral_scanner.scan_passport_mrz(contents)

        # Parse MRZ
        print("📋 Parsing MRZ data...", flush=True)
        parsed_data = MRZParser.parse_mrz(mrz_result["line1"], mrz_result["line2"])

        # Add metadata
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents),
            "file_hash": hashlib.sha256(contents).hexdigest()[:16],
            "scanner": "Mistral AI OCR"
        }

        print(f"✅ Passport scanned successfully: {parsed_data['passport_number']}", flush=True)

        return JSONResponse(content={
            "success": True,
            "data": parsed_data
        })

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan passport: {str(e)}"
        )


@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "message": "Passport Scanner with Mistral AI OCR",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Mistral AI OCR API",
            "Smart MRZ Cropping",
            "ICAO 9303 TD3 Parsing",
            "Checksum Validation",
            "Rate Limiting",
            "File Validation"
        ]
    }


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """
    Webhook endpoint for Telegram bot updates
    This endpoint receives updates from Telegram when running on Railway
    """
    try:
        # Get the update data from Telegram
        update_data = await request.json()

        # Import bot application (lazy import to avoid circular dependency)
        from bot import get_application

        # Get the Telegram application instance
        application = get_application()

        # Process the update
        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)

        return JSONResponse(content={"ok": True})

    except Exception as e:
        print(f"❌ Error processing webhook: {str(e)}", flush=True)
        # Return 200 anyway to avoid Telegram retrying
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
