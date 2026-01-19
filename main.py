"""
Customs Committee Passport MRZ Scanner - Backend API
Uses Customs AI OCR Engine with Anti-Hallucination Guards
"""

import io
import os
import time
import base64
import hashlib
import json
import re
import html
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
    title="Customs Committee Passport Scanner",
    description="Professional MRZ Scanner System",
    version="3.5.0"
)

# CORS middleware
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
# AI SCANNER ENGINE
# ============================================

class CustomsScannerEngine:
    """
    Internal AI OCR API Manager for MRZ Extraction
    """

    def __init__(self, api_key: str):
        # Initialize with 60-second timeout
        self.client = Mistral(
            api_key=api_key,
            timeout_ms=60000
        )
        self.ocr_model = "mistral-ocr-latest"
        self.extraction_model = "mistral-small-latest"
        print(f"🤖 Customs AI Engine initialized", flush=True)

    def scan_passport_mrz(self, image_bytes: bytes, max_retries: int = 3) -> Dict:
        """
        Scan passport MRZ using AI OCR
        """
        print(f"🔍 Starting scan...", flush=True)
        
        attempts = 0
        last_error = None

        while attempts < max_retries:
            try:
                # Convert image to base64
                base64_image = base64.b64encode(image_bytes).decode('utf-8')

                print(f"📤 Sending request to AI Engine (attempt {attempts + 1})...", flush=True)

                ocr_response = self.client.ocr.process(
                    model=self.ocr_model,
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    include_image_base64=True
                )

                if not ocr_response:
                    raise Exception("Empty response from AI Engine")

                print(f"✅ Received response from AI Engine", flush=True)

                # Extract text
                response_text = self._extract_ocr_text(ocr_response)
                
                # Try to parse as JSON first (Fast path)
                try:
                    result = self._parse_as_json(response_text)
                    print(f"✅ Successfully parsed structured data", flush=True)
                except Exception:
                    # Slow path: Use LLM extraction with Anti-Hallucination
                    print(f"⚠️ Raw text mode, using smart extraction...", flush=True)
                    result = self._extract_mrz_with_ai(response_text)
                    print(f"✅ Extracted MRZ using Smart Extraction", flush=True)

                if "error" in result:
                    raise Exception(result["error"])

                # Validate format
                if not self._validate_mrz_format(result):
                    raise Exception("Invalid MRZ format extracted")

                print(f"✅ MRZ extraction successful", flush=True)
                return result

            except Exception as e:
                error_msg = str(e).lower()
                attempts += 1
                print(f"❌ Attempt {attempts} failed: {str(e)[:100]}", flush=True)

                if attempts < max_retries:
                    # Retry logic
                    if "429" in error_msg or "rate limit" in error_msg:
                        time.sleep(3)
                    else:
                        time.sleep(1)
                    continue
                else:
                    last_error = e

        raise HTTPException(
            status_code=500,
            detail=f"Scan failed after {max_retries} attempts: {str(last_error)[:100]}"
        )

    def _extract_ocr_text(self, ocr_response) -> str:
        """Extract clean text from OCR response"""
        try:
            full_text = ""
            if hasattr(ocr_response, 'pages'):
                all_text = [p.markdown for p in ocr_response.pages if hasattr(p, 'markdown')]
                full_text = '\n'.join(all_text)
            
            # Fallbacks
            if not full_text and hasattr(ocr_response, 'content'):
                full_text = str(ocr_response.content)

            # Cleanup
            full_text = html.unescape(full_text)
            full_text = re.sub(r'</[^>]+>', '', full_text)
            full_text = full_text.replace('><', '<<')
            
            # Find lines
            mrz_lines = self._extract_mrz_lines(full_text)
            if len(mrz_lines) >= 2:
                return json.dumps({"line1": mrz_lines[0], "line2": mrz_lines[1]})

            return full_text
        except Exception:
            return str(ocr_response)

    def _smart_split_mrz(self, text: str) -> tuple:
        """
        Intelligently split concatenated MRZ using ROBUST ANCHOR REGEX.
        Fixes the 'IND' issue and incorrect splitting.
        """
        # Sanitize
        text = text.replace(' ', '').replace('\n', '').replace('\r', '')

        # ROBUST REGEX: Finds start of Line 2 based on logic
        # Passport(9) + Check(1) + Nationality(3) + DOB(6) + Check(1)
        line2_anchor = re.compile(r'([A-Z0-9<]{9})(\d)([A-Z<]{3})(\d{6})(\d)', re.IGNORECASE)

        match = line2_anchor.search(text)
        if match:
            split_pos = match.start()
            print(f"🎯 Found Line 2 anchor at pos {split_pos}: {match.group()[:15]}...", flush=True)

            # Line 1 is immediately before Line 2
            # Search backwards for document type P<, I<, V<
            line1_start = -1
            for prefix in ['P<', 'I<', 'A<', 'C<', 'V<']:
                pos = text.rfind(prefix, 0, split_pos)
                if pos != -1:
                    line1_start = pos
                    break
            
            if line1_start != -1:
                line1 = text[line1_start:line1_start+44]
                line2 = text[split_pos:split_pos+44]
            else:
                # Fallback: take 44 chars before split
                start = max(0, split_pos - 44)
                line1 = text[start:split_pos]
                line2 = text[split_pos:split_pos+44]

            # Normalize
            return (self._normalize_mrz_line(line1), self._normalize_mrz_line(line2))

        # Fallback: Blind split
        return (self._normalize_mrz_line(text[:44]), self._normalize_mrz_line(text[44:88]))

    def _extract_mrz_lines(self, text: str) -> list:
        """Extract MRZ lines using Smart Split logic"""
        sanitized = text.replace(' ', '').replace('\n', '').replace('\r', '')
        
        # Check for concatenated string
        if len(sanitized) >= 80 and ('P<' in sanitized.upper()[:10]):
            l1, l2 = self._smart_split_mrz(sanitized)
            return [l1, l2]

        # Line by line
        lines = text.split('\n')
        candidates = []
        for line in lines:
            line = line.strip().replace(' ', '')
            if len(line) >= 80 and 'P<' in line.upper()[:5]:
                l1, l2 = self._smart_split_mrz(line)
                return [l1, l2]
            
            if len(line) == 44 and line.count('<') >= 2:
                candidates.append(line)
            elif line.upper().startswith('P<') and len(line) >= 30:
                candidates.append(line.ljust(44, '<')[:44])

        return candidates

    def _parse_as_json(self, text: str) -> Dict:
        """Parse clean JSON"""
        text = re.sub(r'```[a-z]*\s*', '', text).strip()
        return json.loads(text)

    def _extract_mrz_with_ai(self, ocr_text: str) -> Dict:
        """
        Use AI to extract structured data with ANTI-HALLUCINATION GUARDS
        """
        # 1. PRE-CHECK: Garbage Filter
        if not ocr_text or len(ocr_text) < 30:
            raise Exception("No valid text found in image")
        
        if '<' not in ocr_text and not any(c.isdigit() for c in ocr_text):
            raise Exception("No MRZ patterns found")

        prompt = f"""CRITICAL: Analyze this OCR text.
1. If it does NOT look like a passport MRZ (Must contain 'P<' and '<<' and numbers), return:
   {{"error": "no_mrz_found"}}

2. DO NOT INVENT DATA. Do not return "JOHNSON" or "SPECIMEN" unless it is actually in the text.

3. Extract exactly two lines (44 chars each).

OCR Text:
{ocr_text}

Return JSON:
{{"line1": "TEXT...", "line2": "TEXT..."}}"""

        response = self.client.chat.complete(
            model=self.extraction_model,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        
        # Clean markdown
        content = re.sub(r'```[a-z]*', '', content).strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                
                if "error" in result:
                    raise Exception("AI could not find valid MRZ")

                # ANTI-HALLUCINATION CHECK
                suspicious = ['SPECIMEN', 'SAMPLE', 'JOHNSON', 'JOHNDOE', 'MISTRAL', 'EXAMPLE']
                combined = (result.get('line1', '') + result.get('line2', '')).upper()
                
                for pattern in suspicious:
                    if pattern in combined:
                         raise Exception("Detected fake/sample data - Scan Rejected")

                if "line1" in result and "line2" in result:
                    result["line1"] = self._normalize_mrz_line(result["line1"])
                    result["line2"] = self._normalize_mrz_line(result["line2"])
                    return result
            except json.JSONDecodeError:
                pass
        
        raise Exception("Failed to extract valid MRZ")

    def _normalize_mrz_line(self, line: str) -> str:
        line = html.unescape(line).strip().upper()
        line = line.replace(' ', '').replace('><', '<<')
        if len(line) > 44: line = line[:44]
        elif len(line) < 44: line = line.ljust(44, '<')
        return line

    def _validate_mrz_format(self, result: Dict) -> bool:
        if "line1" not in result or "line2" not in result: return False
        return len(result["line1"]) == 44 and len(result["line2"]) == 44


# ============================================
# STRICT ICAO 9303 MRZ PARSER
# ============================================

class StrictMRZParser:
    """Strict ICAO 9303 TD3 Parser"""

    @staticmethod
    def parse_mrz_strict(line1: str, line2: str) -> Dict:
        print(f"🔍 Parsing MRZ...", flush=True)

        # Basic cleanup
        line1 = line1.strip().upper()
        line2 = line2.strip().upper()

        if len(line1) != 44 or len(line2) != 44:
            raise ValueError("Invalid MRZ line length")

        # LINE 1
        doc_type = line1[0]
        country_code = line1[2:5].replace('<', '')
        names_section = line1[5:44]
        
        surname = names_section.split('<<')[0].replace('<', ' ').strip() if '<<' in names_section else names_section.replace('<', ' ').strip()
        given_names = names_section.split('<<')[1].replace('<', ' ').strip() if '<<' in names_section else ""

        # LINE 2 - FIXED GRID
        passport_raw = line2[0:9]
        passport_check = line2[9]
        nationality_raw = line2[10:13]
        dob_raw = line2[13:19]
        dob_check = line2[19]
        sex_raw = line2[20]
        expiry_raw = line2[21:27]
        expiry_check = line2[27]
        pinfl_raw = line2[28:42]
        pinfl_check = line2[42]

        # ERROR CORRECTION
        # 1. Passport: 2 letters, 7 digits. Fix O->0, 0->O
        pass_clean = passport_raw.replace('<', '')
        if len(pass_clean) >= 2:
            prefix = pass_clean[:2].replace('0', 'O').replace('1', 'I')
            suffix = pass_clean[2:].replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
            passport_number = prefix + suffix
        else:
            passport_number = pass_clean

        # 2. Nationality
        nationality = nationality_raw.replace('<', '')
        if nationality in ['ZBO', 'LZB', 'USB', 'U2B', 'UZ8', 'O2B']: nationality = 'UZB'
        
        # Force UZB if passport starts with known prefix
        if nationality != 'UZB' and passport_number[:2] in ['FA', 'FB', 'FC', 'FD', 'AC', 'AD', 'AA']:
            nationality = 'UZB'

        # 3. Dates/PINFL: Fix O->0
        dob = dob_raw.replace('O', '0')
        expiry = expiry_raw.replace('O', '0')
        pinfl = pinfl_raw.replace('O', '0').replace('<', '')
        sex = sex_raw.replace('<', '')

        # VALIDATION
        validations = {
            "passport_valid": StrictMRZParser.validate(passport_raw, passport_check),
            "dob_valid": StrictMRZParser.validate(dob_raw, dob_check),
            "expiry_valid": StrictMRZParser.validate(expiry_raw, expiry_check),
            "pinfl_valid": StrictMRZParser.validate(pinfl_raw, pinfl_check)
        }
        
        all_valid = all(validations.values())

        return {
            "passport_number": passport_number,
            "surname": surname,
            "name": given_names,
            "given_names": given_names,
            "birth_date": StrictMRZParser.fmt_date(dob),
            "date_of_birth": StrictMRZParser.fmt_date(dob),
            "expiry_date": StrictMRZParser.fmt_date(expiry),
            "date_of_expiry": StrictMRZParser.fmt_date(expiry),
            "sex": sex,
            "nationality": nationality,
            "pinfl": pinfl,
            "personal_number": pinfl,
            "document_type": doc_type,
            "country_code": country_code,
            "validation_status": "PASS" if all_valid else "WARNING",
            "raw_mrz": {"line1": line1, "line2": line2}
        }

    @staticmethod
    def fmt_date(yymmdd):
        if len(yymmdd) != 6 or not yymmdd.isdigit(): return yymmdd
        return f"{yymmdd[4:6]}.{yymmdd[2:4]}.{'20'+yymmdd[0:2] if int(yymmdd[0:2])<50 else '19'+yymmdd[0:2]}"

    @staticmethod
    def validate(data, check):
        if not check.isdigit(): return False
        weights = [7, 3, 1]
        total = 0
        for i, c in enumerate(data):
            val = int(c) if c.isdigit() else (ord(c)-55 if c.isalpha() else 0)
            total += val * weights[i % 3]
        return (total % 10) == int(check)


# ============================================
# SECURITY & CONFIG
# ============================================

request_timestamps = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def check_rate_limit(client_id: str) -> bool:
    current_time = time.time()
    if client_id not in request_timestamps: request_timestamps[client_id] = []
    request_timestamps[client_id] = [ts for ts in request_timestamps[client_id] if current_time - ts < RATE_LIMIT_WINDOW]
    if len(request_timestamps[client_id]) >= MAX_REQUESTS_PER_WINDOW: return False
    request_timestamps[client_id].append(current_time)
    return True

def validate_image_file(file: UploadFile, contents: bytes) -> None:
    if len(contents) > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail="File too large")
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS: raise HTTPException(status_code=400, detail="Invalid file type")
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except: raise HTTPException(status_code=400, detail="Invalid image")


# ============================================
# API ENDPOINTS
# ============================================

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "JWSVnIJhbnyhc80PY32AhKkxEbS4SFFi")
scanner_engine = CustomsScannerEngine(api_key=MISTRAL_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "customs-passport-scanner",
        "version": "3.5.0"
    }

@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """Main scanning endpoint"""
    try:
        client_id = request.client.host if request.client else "unknown"
        if not check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Too many requests")

        contents = await file.read()
        if not contents: raise HTTPException(status_code=400, detail="Empty file")
        validate_image_file(file, contents)

        print(f"📸 Processing image: {file.filename}", flush=True)

        # 1. Scan (OCR)
        mrz_result = scanner_engine.scan_passport_mrz(contents)

        # 2. Parse (Strict Logic)
        parsed_data = StrictMRZParser.parse_mrz_strict(mrz_result["line1"], mrz_result["line2"])

        # 3. Add Metadata (BRANDING REMOVED)
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "scanner": "Customs AI Scanner (v3.5)" # <--- NEW NAME
        }

        print(f"✅ Scan completed: {parsed_data['passport_number']}", flush=True)
        return JSONResponse(content={"success": True, "data": parsed_data})

    except HTTPException: raise
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        update_data = await request.json()
        from bot import get_application
        application = get_application()
        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)
        return JSONResponse(content={"ok": True})
    except Exception as e:
        print(f"❌ Webhook error: {str(e)}", flush=True)
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
