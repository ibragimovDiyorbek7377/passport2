"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BOJXONA PASSPORT SCANNER - 100% LOKAL VERSIYA                      ║
║                                                                              ║
║  ✅ API YO'Q - To'liq offline ishlaydi                                       ║
║  ✅ Tesseract OCR - Lokal OCR engine                                         ║
║  ✅ ICAO 9303 TD3 - Xalqaro standart                                         ║
║  ✅ Xavfsiz - Ma'lumotlar serverga yuborilmaydi                              ║
║                                                                              ║
║  Versiya: 2.0.0 (Lokal)                                                      ║
║  Standart: ICAO Doc 9303 Part 4                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import io
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Image Processing
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# OCR (Lokal)
import pytesseract

# OpenCV (Lokal image preprocessing)
import cv2


# ============================================
# KONFIGURATSIYA
# ============================================

class Config:
    """Tizim sozlamalari"""
    VERSION = "2.0.0"
    SERVICE_NAME = "Bojxona Passport Scanner (Lokal)"

    # Xavfsizlik
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    RATE_LIMIT_WINDOW = 60  # sekund
    MAX_REQUESTS_PER_WINDOW = 30

    # OCR sozlamalari - MRZ uchun optimallashtirilgan
    TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    TESSERACT_LANG = 'eng'

    # MRZ parametrlari (ICAO 9303 TD3)
    MRZ_LINE_LENGTH = 44
    MRZ_TOTAL_LINES = 2


# ============================================
# XAVFSIZLIK MODULI
# ============================================

class SecurityModule:
    """Rate limiting va file validation"""

    def __init__(self):
        self._request_timestamps: Dict[str, List[float]] = {}

    def check_rate_limit(self, client_id: str) -> bool:
        """Rate limiting - DDoS himoyasi"""
        current_time = time.time()

        if client_id not in self._request_timestamps:
            self._request_timestamps[client_id] = []

        self._request_timestamps[client_id] = [
            ts for ts in self._request_timestamps[client_id]
            if current_time - ts < Config.RATE_LIMIT_WINDOW
        ]

        if len(self._request_timestamps[client_id]) >= Config.MAX_REQUESTS_PER_WINDOW:
            return False

        self._request_timestamps[client_id].append(current_time)
        return True

    def validate_image(self, file: UploadFile, contents: bytes) -> None:
        """Rasm validatsiyasi"""

        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Fayl juda katta (max 10MB)")

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Bo'sh fayl")

        if file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in Config.ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Noto'g'ri format: {ext}")

        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Noto'g'ri rasm")


# ============================================
# RASM PREPROCESSING (OpenCV)
# ============================================

class ImagePreprocessor:
    """Rasmni OCR uchun tayyorlash"""

    @staticmethod
    def preprocess_for_mrz(image_bytes: bytes) -> np.ndarray:
        """MRZ zonasini OCR uchun tayyorlash"""

        # PIL dan numpy array ga
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_array = np.array(pil_image)

        # Grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Resize (Tesseract uchun optimal)
        height, width = gray.shape
        if width < 1000:
            scale = 1000 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary

    @staticmethod
    def extract_mrz_region(img_array: np.ndarray) -> np.ndarray:
        """MRZ zonasini kesish (pastki 25%)"""
        height = img_array.shape[0]
        mrz_height = int(height * 0.25)
        return img_array[height - mrz_height:, :]


# ============================================
# LOKAL OCR ENGINE (Tesseract)
# ============================================

class LocalOCREngine:
    """Tesseract OCR - 100% Lokal"""

    def __init__(self):
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR v{version} initialized", flush=True)
        except Exception as e:
            raise RuntimeError(f"Tesseract topilmadi: {e}")

    def extract_text(self, image: np.ndarray) -> str:
        """Rasmdan matn ajratish"""
        return pytesseract.image_to_string(
            image,
            lang=Config.TESSERACT_LANG,
            config=Config.TESSERACT_CONFIG
        )

    def extract_mrz_lines(self, image_bytes: bytes) -> Tuple[str, str]:
        """MRZ qatorlarini ajratish"""
        print("🔍 MRZ qatorlarini qidirish...", flush=True)

        # Preprocessing
        processed = ImagePreprocessor.preprocess_for_mrz(image_bytes)

        # MRZ zonasini kesish
        mrz_region = ImagePreprocessor.extract_mrz_region(processed)

        # OCR
        raw_text = self.extract_text(mrz_region)

        # MRZ qatorlarini topish
        lines = self._find_mrz_lines(raw_text)

        # Agar topilmasa, to'liq rasmdan qayta urinish
        if len(lines) < 2:
            raw_text = self.extract_text(processed)
            lines = self._find_mrz_lines(raw_text)

        if len(lines) < 2:
            raise ValueError("MRZ topilmadi. Aniqroq rasm yuboring.")

        line1 = self._normalize_line(lines[0])
        line2 = self._normalize_line(lines[1])

        print(f"✅ MRZ topildi", flush=True)
        print(f"   Line1: {line1}", flush=True)
        print(f"   Line2: {line2}", flush=True)

        return line1, line2

    def _find_mrz_lines(self, text: str) -> List[str]:
        """Matndan MRZ qatorlarini ajratish"""
        candidates = []

        for line in text.split('\n'):
            cleaned = self._clean_line(line)

            if len(cleaned) >= 40:
                # P< bilan boshlansa - Line 1
                if cleaned.startswith('P') or 'P<' in cleaned[:5]:
                    candidates.insert(0, cleaned)
                # Raqamlar ko'p bo'lsa - Line 2
                elif sum(c.isdigit() for c in cleaned) >= 10:
                    candidates.append(cleaned)
                elif len(cleaned) >= 44:
                    candidates.append(cleaned)

        # Concatenated MRZ (88 belgi)
        full_text = ''.join(text.split())
        full_text = self._clean_line(full_text)
        if len(full_text) >= 88 and 'P<' in full_text[:10]:
            return self._smart_split(full_text)

        return candidates[:2]

    def _smart_split(self, text: str) -> List[str]:
        """88 belgini 2 qatorga ajratish"""
        # Line 2 boshlanishini topish (Passport + Check + Nationality pattern)
        pattern = r'([A-Z]{2}\d{7})(\d)([A-Z]{3})(\d{6})(\d)'
        match = re.search(pattern, text)

        if match:
            split_pos = match.start()
            line1 = text[:44] if split_pos < 44 else text[:split_pos]
            line2 = text[split_pos:split_pos+44]
        else:
            line1 = text[:44]
            line2 = text[44:88]

        return [line1, line2]

    def _clean_line(self, line: str) -> str:
        """Faqat MRZ belgilarini qoldirish"""
        allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
        return ''.join(c for c in line.upper() if c in allowed)

    def _normalize_line(self, line: str) -> str:
        """44 belgiga moslashtirish"""
        line = self._clean_line(line)
        if len(line) > 44:
            line = line[:44]
        elif len(line) < 44:
            line = line.ljust(44, '<')
        return line


# ============================================
# ICAO 9303 TD3 MRZ PARSER
# ============================================

@dataclass
class MRZValidation:
    """Validatsiya natijasi"""
    passport_valid: bool
    dob_valid: bool
    expiry_valid: bool
    personal_number_valid: bool
    composite_valid: bool
    all_valid: bool


class ICAO9303Parser:
    """
    ICAO 9303 TD3 MRZ Parser

    TD3 Format (Passport - 2 qator, har biri 44 belgi):

    Line 1: P<UTOSURNAME<<GIVEN<NAMES<<<<<<<<<<<<<<<<<<
            [0]    = Document type (P)
            [1]    = Document subtype
            [2:5]  = Issuing country (3 belgi)
            [5:44] = Names (Familiya<<Ism)

    Line 2: L898902C<3UTO6908061F9406236ZE184226B<<<<<14
            [0:9]   = Passport number (9 belgi)
            [9]     = Passport check digit
            [10:13] = Nationality (3 belgi)
            [13:19] = Date of birth YYMMDD
            [19]    = DOB check digit
            [20]    = Sex (M/F/<)
            [21:27] = Expiry date YYMMDD
            [27]    = Expiry check digit
            [28:42] = Personal number (14 belgi) - JSHSHIR
            [42]    = Personal number check digit
            [43]    = Composite check digit
    """

    # ICAO 9303 checksum weights
    WEIGHTS = [7, 3, 1]

    # O'zbekiston pasport prefikslari
    UZB_PREFIXES = {'AA', 'AB', 'AC', 'AD', 'FA', 'FB', 'FC', 'FD', 'FK', 'HA', 'HB'}

    @classmethod
    def parse(cls, line1: str, line2: str) -> Dict:
        """MRZ qatorlarini parse qilish"""

        print(f"🔍 ICAO 9303 TD3 parsing...", flush=True)

        # Uzunlik tekshiruvi
        if len(line1) != 44 or len(line2) != 44:
            raise ValueError(f"MRZ uzunligi noto'g'ri: L1={len(line1)}, L2={len(line2)}")

        # ==========================================
        # LINE 1 PARSING
        # ==========================================
        doc_type = line1[0]                    # P
        doc_subtype = line1[1]                 # <
        issuing_country = line1[2:5].replace('<', '')  # UZB
        names_section = line1[5:44]            # ALLAMOV<<OTABEK<KHUDAYBERGANOVICH<<<<<

        # Ism va familiyani ajratish
        surname, given_names = cls._parse_names(names_section)

        # ==========================================
        # LINE 2 PARSING (ICAO 9303 TD3 POZITSIYALAR)
        # ==========================================
        passport_raw = line2[0:9]              # FK0005527
        passport_check = line2[9]              # 8
        nationality_raw = line2[10:13]         # UZB
        dob_raw = line2[13:19]                 # 901107
        dob_check = line2[19]                  # 8
        sex = line2[20]                        # M
        expiry_raw = line2[21:27]              # 290430
        expiry_check = line2[27]               # 8
        personal_num_raw = line2[28:42]        # 30711903330048
        personal_num_check = line2[42]         # 2
        composite_check = line2[43]            # 6

        # ==========================================
        # OCR XATOLARINI TUZATISH
        # ==========================================
        passport_number = cls._fix_passport_number(passport_raw)
        nationality = cls._fix_nationality(nationality_raw, passport_number)
        dob = cls._fix_digits(dob_raw)
        expiry = cls._fix_digits(expiry_raw)
        personal_number = cls._fix_digits(personal_num_raw.replace('<', ''))
        sex = cls._fix_sex(sex)

        # ==========================================
        # ICAO 9303 CHECKSUM VALIDATSIYASI
        # ==========================================
        validation = cls._validate_all_checksums(
            passport_raw, passport_check,
            dob_raw, dob_check,
            expiry_raw, expiry_check,
            personal_num_raw, personal_num_check,
            composite_check
        )

        # ==========================================
        # NATIJA
        # ==========================================
        result = {
            # Asosiy ma'lumotlar
            "passport_number": passport_number,
            "surname": surname,
            "given_names": given_names,
            "name": given_names,
            "date_of_birth": cls._format_date(dob),
            "birth_date": cls._format_date(dob),
            "sex": sex,
            "date_of_expiry": cls._format_date(expiry),
            "expiry_date": cls._format_date(expiry),
            "personal_number": personal_number,
            "pinfl": personal_number,
            "nationality": nationality,

            # Hujjat ma'lumotlari
            "document_type": doc_type,
            "document_subtype": doc_subtype,
            "issuing_country": issuing_country,

            # Validatsiya
            "validation_status": "PASS" if validation.all_valid else "WARNING",
            "validations": {
                "passport_valid": validation.passport_valid,
                "dob_valid": validation.dob_valid,
                "expiry_valid": validation.expiry_valid,
                "personal_number_valid": validation.personal_number_valid,
                "composite_valid": validation.composite_valid,
                "all_valid": validation.all_valid
            },

            # Raw MRZ
            "raw_mrz": {
                "line1": line1,
                "line2": line2
            }
        }

        print(f"✅ Parse muvaffaqiyatli: {passport_number}", flush=True)
        return result

    # ==========================================
    # YORDAMCHI METODLAR
    # ==========================================

    @classmethod
    def _parse_names(cls, names_section: str) -> Tuple[str, str]:
        """Familiya va ismni ajratish"""
        if '<<' in names_section:
            parts = names_section.split('<<', 1)
            surname = parts[0].replace('<', ' ').strip()
            given_names = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ''
        else:
            surname = names_section.replace('<', ' ').strip()
            given_names = ''
        return surname, given_names

    @classmethod
    def _fix_passport_number(cls, raw: str) -> str:
        """Pasport raqamini tuzatish (2 harf + 7 raqam)"""
        clean = raw.replace('<', '')
        if len(clean) < 2:
            return clean

        # Birinchi 2 ta = harflar
        prefix = clean[:2].replace('0', 'O').replace('1', 'I')
        # Qolgan 7 ta = raqamlar
        suffix = clean[2:].replace('O', '0').replace('I', '1').replace('l', '1')

        return prefix + suffix

    @classmethod
    def _fix_nationality(cls, raw: str, passport_number: str) -> str:
        """Nationality ni tuzatish"""
        nationality = raw.replace('<', '')

        # OCR xatolarini tuzatish
        fixes = {
            'ZBO': 'UZB', 'LZB': 'UZB', 'USB': 'UZB',
            'U2B': 'UZB', 'UZ8': 'UZB', 'O2B': 'UZB',
            'UZ0': 'UZB', '0ZB': 'UZB'
        }
        nationality = fixes.get(nationality, nationality)

        # O'zbekiston pasporti prefiksini tekshirish
        if passport_number[:2] in cls.UZB_PREFIXES and nationality != 'UZB':
            nationality = 'UZB'

        return nationality

    @classmethod
    def _fix_digits(cls, raw: str) -> str:
        """Raqamlarni tuzatish (O->0, I->1)"""
        return raw.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')

    @classmethod
    def _fix_sex(cls, raw: str) -> str:
        """Jinsni tuzatish"""
        raw = raw.upper()
        if raw in ('M', 'F'):
            return raw
        return 'X'  # Noma'lum

    @classmethod
    def _format_date(cls, yymmdd: str) -> str:
        """YYMMDD -> DD.MM.YYYY"""
        if len(yymmdd) != 6 or not yymmdd.isdigit():
            return yymmdd

        yy = int(yymmdd[0:2])
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]

        # ICAO 9303: < 50 = 2000s, >= 50 = 1900s
        year = f"20{yy:02d}" if yy < 50 else f"19{yy:02d}"

        return f"{dd}.{mm}.{year}"

    # ==========================================
    # CHECKSUM METODLARI (ICAO 9303 Modulo 10)
    # ==========================================

    @classmethod
    def _calculate_checksum(cls, data: str) -> int:
        """
        ICAO 9303 Modulo 10 checksum
        Algorithm: Sum of (value × weight) mod 10
        Weights: 7, 3, 1 (repeating)
        Values: 0-9 = 0-9, A-Z = 10-35, < = 0
        """
        total = 0
        for i, char in enumerate(data):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - 55  # A=10, B=11, ..., Z=35
            else:
                value = 0  # < va boshqa belgilar

            total += value * cls.WEIGHTS[i % 3]

        return total % 10

    @classmethod
    def _validate_checksum(cls, data: str, check_digit: str) -> bool:
        """Bitta checksum tekshirish"""
        if not check_digit.isdigit():
            return False
        return cls._calculate_checksum(data) == int(check_digit)

    @classmethod
    def _validate_all_checksums(
        cls,
        passport_raw: str, passport_check: str,
        dob_raw: str, dob_check: str,
        expiry_raw: str, expiry_check: str,
        personal_raw: str, personal_check: str,
        composite_check: str
    ) -> MRZValidation:
        """Barcha checksumlarni tekshirish"""

        passport_valid = cls._validate_checksum(passport_raw, passport_check)
        dob_valid = cls._validate_checksum(dob_raw, dob_check)
        expiry_valid = cls._validate_checksum(expiry_raw, expiry_check)
        personal_valid = cls._validate_checksum(personal_raw, personal_check)

        # Composite checksum
        composite_data = (
            passport_raw + passport_check +
            dob_raw + dob_check +
            expiry_raw + expiry_check +
            personal_raw + personal_check
        )
        composite_valid = cls._validate_checksum(composite_data, composite_check)

        all_valid = all([
            passport_valid, dob_valid, expiry_valid,
            personal_valid, composite_valid
        ])

        return MRZValidation(
            passport_valid=passport_valid,
            dob_valid=dob_valid,
            expiry_valid=expiry_valid,
            personal_number_valid=personal_valid,
            composite_valid=composite_valid,
            all_valid=all_valid
        )


# ============================================
# ASOSIY SCANNER CLASS
# ============================================

class LocalPassportScanner:
    """To'liq lokal pasport scanner"""

    def __init__(self):
        self.security = SecurityModule()
        self.ocr = LocalOCREngine()
        print(f"✅ {Config.SERVICE_NAME} v{Config.VERSION} tayyor", flush=True)

    def scan(self, image_bytes: bytes, filename: str = "unknown") -> Dict:
        """Pasportni skanerlash"""
        start_time = time.time()

        try:
            # 1. OCR - MRZ qatorlarini ajratish
            line1, line2 = self.ocr.extract_mrz_lines(image_bytes)

            # 2. Parse - ICAO 9303 bo'yicha
            parsed_data = ICAO9303Parser.parse(line1, line2)

            # 3. Metadata
            scan_time = time.time() - start_time
            parsed_data["scan_metadata"] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "file_name": filename,
                "scan_time_ms": int(scan_time * 1000),
                "scanner": f"{Config.SERVICE_NAME} v{Config.VERSION}",
                "engine": "Tesseract OCR (Lokal)",
                "api_used": False
            }

            return parsed_data

        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Xato: {str(e)}")


# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Bojxona Passport Scanner",
    description="100% Lokal MRZ Scanner - ICAO 9303 TD3",
    version=Config.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Global instances
scanner = LocalPassportScanner()
security = SecurityModule()


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Mini App sahifasi"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": Config.SERVICE_NAME,
        "version": Config.VERSION,
        "engine": "Tesseract OCR (Lokal)",
        "api_required": False
    }


@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Pasportni skanerlash

    🔒 100% LOKAL - Ma'lumotlar hech qayerga yuborilmaydi
    """
    try:
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"
        if not security.check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Ko'p so'rov. 1 daqiqa kuting.")

        # Faylni o'qish
        contents = await file.read()

        # Validatsiya
        security.validate_image(file, contents)

        print(f"📸 Skanerlash: {file.filename}", flush=True)

        # Skanerlash
        result = scanner.scan(contents, file.filename)

        print(f"✅ Tayyor: {result.get('passport_number')}", flush=True)

        return JSONResponse(content={"success": True, "data": result})

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Xato: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Telegram webhook"""
    try:
        from bot import get_application
        from telegram import Update

        update_data = await request.json()
        application = get_application()
        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)

        return JSONResponse(content={"ok": True})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=200)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         BOJXONA PASSPORT SCANNER - LOKAL VERSIYA                 ║
║                                                                  ║
║  🔒 API YO'Q - Barcha ma'lumotlar lokal qayta ishlanadi         ║
║  ✅ Tesseract OCR - Offline ishlaydi                            ║
║  ✅ ICAO 9303 TD3 - Xalqaro standart                            ║
║                                                                  ║
║  Port: {port}                                                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
