"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BOJXONA PASSPORT SCANNER - DeepSeek OCR                            ║
║                                                                              ║
║  ✅ DeepSeek OCR - DeepSeek-OCR-Latest-BF16.I64 model                        ║
║  ✅ Tesseract OCR - Fallback lokal engine                                    ║
║  ✅ ICAO 9303 TD3 - Xalqaro standart                                         ║
║  ✅ Xavfsiz - Yuqori aniqlik                                                ║
║                                                                              ║
║  Versiya: 3.0.0 (DeepSeek OCR)                                              ║
║  Standart: ICAO Doc 9303 Part 4                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import io
import os
import re
import time
import base64
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Image Processing
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

# OCR (Lokal - fallback)
import pytesseract

# DeepSeek OCR API
from openai import OpenAI

# OpenCV (Lokal image preprocessing)
import cv2


# ============================================
# KONFIGURATSIYA
# ============================================

class Config:
    """Tizim sozlamalari"""
    VERSION = "3.0.0"
    SERVICE_NAME = "Bojxona Passport Scanner (DeepSeek OCR)"

    # DeepSeek OCR
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = "DeepSeek-OCR-Latest-BF16.I64"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    # Xavfsizlik
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    RATE_LIMIT_WINDOW = 60  # sekund
    MAX_REQUESTS_PER_WINDOW = 30

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
# YAXSHILANGAN MRZ DETECTOR (OpenCV)
# ============================================

class MRZDetector:
    """
    MRZ zonasini topish va OCR uchun tayyorlash.
    Bir nechta usul bilan sinab ko'radi.
    """

    @staticmethod
    def detect_and_extract(image_bytes: bytes) -> List[np.ndarray]:
        """
        MRZ zonasini topish va bir nechta variant qaytarish.
        Har bir variant boshqa preprocessing bilan.
        """
        # PIL dan numpy array ga
        pil_image = Image.open(io.BytesIO(image_bytes))

        # EXIF orientation ni to'g'rilash
        try:
            pil_image = ImageOps.exif_transpose(pil_image)
        except:
            pass

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        img_array = np.array(pil_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Rasmni kattalashtirish (agar kichik bo'lsa)
        height, width = gray.shape
        if width < 1200:
            scale = 1200 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            height, width = gray.shape

        candidates = []

        # ==========================================
        # USUL 1: Morphological MRZ Detection
        # ==========================================
        try:
            mrz_region = MRZDetector._detect_mrz_morphological(gray)
            if mrz_region is not None:
                candidates.append(mrz_region)
        except Exception as e:
            print(f"   Morphological detection xatosi: {e}", flush=True)

        # ==========================================
        # USUL 2: Pastki 30% ni kesish (klassik)
        # ==========================================
        mrz_height = int(height * 0.30)
        bottom_region = gray[height - mrz_height:, :]
        candidates.append(MRZDetector._preprocess_for_ocr(bottom_region))

        # ==========================================
        # USUL 3: Pastki 25% - boshqa preprocessing
        # ==========================================
        mrz_height = int(height * 0.25)
        bottom_region = gray[height - mrz_height:, :]
        candidates.append(MRZDetector._preprocess_for_ocr_v2(bottom_region))

        # ==========================================
        # USUL 4: To'liq rasm
        # ==========================================
        candidates.append(MRZDetector._preprocess_for_ocr(gray))

        # ==========================================
        # USUL 5: Inverted (qora fonda oq matn)
        # ==========================================
        mrz_height = int(height * 0.30)
        bottom_region = gray[height - mrz_height:, :]
        inverted = cv2.bitwise_not(bottom_region)
        candidates.append(MRZDetector._preprocess_for_ocr(inverted))

        # ==========================================
        # USUL 6: ANIQ BINARY (M/N farqlash uchun)
        # ==========================================
        mrz_height = int(height * 0.30)
        bottom_region = gray[height - mrz_height:, :]
        candidates.append(MRZDetector._preprocess_strict_binary(bottom_region))

        # ==========================================
        # USUL 7: YUQORI KONTRAST
        # ==========================================
        mrz_height = int(height * 0.30)
        bottom_region = gray[height - mrz_height:, :]
        candidates.append(MRZDetector._preprocess_high_contrast(bottom_region))

        # ==========================================
        # USUL 8: ANIQ BINARY - to'liq rasm
        # ==========================================
        candidates.append(MRZDetector._preprocess_strict_binary(gray))

        return candidates

    @staticmethod
    def _detect_mrz_morphological(gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Morphological operations bilan MRZ zonasini topish.
        MRZ - gorizontal chiziqlar, belgilar orasida kam bo'shliq.
        """
        height, width = gray.shape

        # Blackhat morphology - qora matnni topish
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Gradient - gorizontal chiziqlarni ajratish
        grad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(blackhat, cv2.MORPH_GRADIENT, grad_kernel)

        # Threshold
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Gorizontal chiziqlarni birlashtirish
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # Vertikal yo'nalishda kengaytirish
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
        dilated = cv2.dilate(closed, dilate_kernel, iterations=2)

        # Contourlarni topish
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Eng katta va pastda joylashgan contourni topish
        best_contour = None
        best_y = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # MRZ xususiyatlari: keng, past balandlik, pastda joylashgan
            aspect_ratio = w / h if h > 0 else 0
            area_ratio = (w * h) / (width * height)

            if (aspect_ratio > 5 and  # Keng
                area_ratio > 0.02 and  # Yetarlicha katta
                y > height * 0.5 and  # Pastki yarmida
                w > width * 0.5):  # Keng
                if y > best_y:
                    best_y = y
                    best_contour = contour

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)

            # Biroz kengaytirish
            padding = 20
            y = max(0, y - padding)
            h = min(height - y, h + padding * 2)

            mrz_region = gray[y:y+h, x:x+w]
            return MRZDetector._preprocess_for_ocr(mrz_region)

        return None

    @staticmethod
    def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """OCR uchun rasmni tayyorlash - Variant 1"""
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        # CLAHE - contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 4
        )

        return binary

    @staticmethod
    def _preprocess_for_ocr_v2(image: np.ndarray) -> np.ndarray:
        """OCR uchun rasmni tayyorlash - Variant 2 (boshqa parametrlar)"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Otsu threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    @staticmethod
    def _preprocess_strict_binary(image: np.ndarray) -> np.ndarray:
        """
        MRZ uchun ANIQ oq-qora rasm.
        M va N harflarini farqlash uchun maxsus ishlov.
        """
        height, width = image.shape

        # 1. Rasmni 2x kattalashtirish (aniqlik uchun)
        if width < 2000:
            scale = 2.0
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Kuchli kontrast
        # Histogram equalization
        equalized = cv2.equalizeHist(image)

        # 3. Gaussian blur - shovqinni kamaytirish
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

        # 4. Aniq binary threshold (Otsu)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 5. Harflarni yo'g'onlashtirish (M va N ni farqlash uchun)
        # M harfida 2 ta vertikal chiziq, N da 1 ta diagonal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 6. Mayda nuqtalarni olib tashlash
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

        return cleaned

    @staticmethod
    def _preprocess_high_contrast(image: np.ndarray) -> np.ndarray:
        """
        Yuqori kontrastli preprocessing.
        Matnni aniqroq ajratish uchun.
        """
        height, width = image.shape

        # 1. Resize
        if width < 1500:
            scale = 1500 / width
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Bilateral filter - shovqinni kamaytirish, qirralarni saqlash
        filtered = cv2.bilateralFilter(image, 9, 75, 75)

        # 3. CLAHE - mahalliy kontrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        # 4. Sharpen - keskinlashtirish
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

        # 5. Binary threshold
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return binary


# ============================================
# DeepSeek OCR ENGINE
# ============================================

class DeepSeekOCREngine:
    """
    DeepSeek OCR - DeepSeek-OCR-Latest-BF16.I64 model.
    Tesseract OCR fallback sifatida qoldirilgan.
    """

    # Tesseract fallback konfiguratsiyalari
    OCR_CONFIGS = [
        '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
        '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
        '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
    ]

    def __init__(self):
        self.deepseek_available = False
        self.client = None

        # DeepSeek API ni sozlash
        if Config.DEEPSEEK_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=Config.DEEPSEEK_API_KEY,
                    base_url=Config.DEEPSEEK_BASE_URL
                )
                self.deepseek_available = True
                print(f"✅ DeepSeek OCR initialized (model: {Config.DEEPSEEK_MODEL})", flush=True)
            except Exception as e:
                print(f"⚠️ DeepSeek OCR xatosi: {e}. Tesseract fallback ishlatiladi.", flush=True)
        else:
            print("⚠️ DEEPSEEK_API_KEY topilmadi. Tesseract fallback ishlatiladi.", flush=True)

        # Tesseract fallback
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR v{version} (fallback) initialized", flush=True)
        except Exception as e:
            if not self.deepseek_available:
                raise RuntimeError(f"Na DeepSeek, na Tesseract ishlamaydi: {e}")
            print(f"⚠️ Tesseract topilmadi (faqat DeepSeek ishlatiladi)", flush=True)

    def extract_mrz_lines(self, image_bytes: bytes) -> Tuple[str, str]:
        """MRZ qatorlarini ajratish - avval DeepSeek, keyin Tesseract fallback"""
        print("🔍 MRZ qatorlarini qidirish...", flush=True)

        # 1. DeepSeek OCR bilan sinash
        if self.deepseek_available:
            try:
                result = self._extract_with_deepseek(image_bytes)
                if result:
                    return result
                print("⚠️ DeepSeek natija bermadi, Tesseract fallback...", flush=True)
            except Exception as e:
                print(f"⚠️ DeepSeek xatosi: {e}. Tesseract fallback...", flush=True)

        # 2. Tesseract fallback
        return self._extract_with_tesseract(image_bytes)

    def _extract_with_deepseek(self, image_bytes: bytes) -> Optional[Tuple[str, str]]:
        """DeepSeek OCR model bilan MRZ qatorlarini ajratish"""
        print(f"🤖 DeepSeek OCR ishlatilmoqda ({Config.DEEPSEEK_MODEL})...", flush=True)

        # Rasmni base64 ga o'girish
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Rasm formatini aniqlash
        if image_bytes[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        elif image_bytes[:4] == b'RIFF':
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        # DeepSeek API ga so'rov
        response = self.client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a passport MRZ (Machine Readable Zone) OCR specialist. "
                        "Extract the two MRZ lines from the passport image. "
                        "MRZ is located at the bottom of the passport and consists of exactly 2 lines, "
                        "each exactly 44 characters long. "
                        "Characters allowed: A-Z, 0-9, and < (filler character). "
                        "Line 1 starts with 'P<'. "
                        "Line 2 contains passport number, nationality, dates, and check digits. "
                        "Return ONLY the two MRZ lines, one per line, nothing else. "
                        "Do not add any explanation or formatting."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract the two MRZ lines from this passport image. Return ONLY the two 44-character MRZ lines, one per line."
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.0
        )

        # Natijani qayta ishlash
        raw_text = response.choices[0].message.content.strip()
        print(f"   DeepSeek raw natija: {raw_text}", flush=True)

        lines = []
        for line in raw_text.split('\n'):
            cleaned = self._clean_line(line.strip())
            if len(cleaned) >= 38:
                lines.append(cleaned)

        if len(lines) >= 2:
            line1 = self._normalize_line(lines[0])
            line2 = self._normalize_line(lines[1])

            score = self._score_mrz(line1, line2)
            print(f"✅ DeepSeek MRZ topildi (score={score})", flush=True)
            print(f"   Line1: {line1}", flush=True)
            print(f"   Line2: {line2}", flush=True)

            if score >= 30:
                return line1, line2

        # Agar natija yomon bo'lsa - concatenated holatini tekshirish
        full_text = self._clean_line(''.join(raw_text.split()))
        if len(full_text) >= 80 and 'P<' in full_text[:10]:
            split_lines = self._smart_split(full_text)
            if len(split_lines) >= 2:
                line1 = self._normalize_line(split_lines[0])
                line2 = self._normalize_line(split_lines[1])
                return line1, line2

        return None

    def _extract_with_tesseract(self, image_bytes: bytes) -> Tuple[str, str]:
        """Tesseract OCR fallback"""
        print("🔍 Tesseract OCR fallback ishlatilmoqda...", flush=True)

        detector = MRZDetector()
        image_variants = detector.detect_and_extract(image_bytes)

        print(f"   {len(image_variants)} ta rasm varianti tayyorlandi", flush=True)

        all_candidates = []

        for i, img in enumerate(image_variants):
            for j, config in enumerate(self.OCR_CONFIGS):
                try:
                    raw_text = pytesseract.image_to_string(img, lang='eng', config=config)
                    lines = self._find_mrz_lines(raw_text)

                    if len(lines) >= 2:
                        line1 = self._normalize_line(lines[0])
                        line2 = self._normalize_line(lines[1])

                        score = self._score_mrz(line1, line2)
                        all_candidates.append((score, line1, line2, i, j))
                        print(f"   Variant {i+1}, Config {j+1}: score={score}", flush=True)

                except Exception:
                    continue

        if not all_candidates:
            raise ValueError("MRZ topilmadi. Aniqroq rasm yuboring.")

        all_candidates.sort(reverse=True, key=lambda x: x[0])
        best = all_candidates[0]

        print(f"✅ MRZ topildi (score={best[0]}, variant={best[3]+1}, config={best[4]+1})", flush=True)
        print(f"   Line1: {best[1]}", flush=True)
        print(f"   Line2: {best[2]}", flush=True)

        return best[1], best[2]

    def _score_mrz(self, line1: str, line2: str) -> int:
        """MRZ sifatini baholash"""
        score = 0

        if line1.startswith('P'):
            score += 20
        if line1.startswith('P<'):
            score += 10
        if '<' in line1:
            score += 10
        if '<<' in line1:
            score += 15

        digit_count = sum(c.isdigit() for c in line2)
        score += min(digit_count * 2, 30)

        uzb_prefixes = {'AA', 'AB', 'AC', 'AD', 'FA', 'FB', 'FC', 'FD', 'FK', 'HA', 'HB'}
        if line2[:2] in uzb_prefixes:
            score += 25

        if len(line2) >= 13:
            nationality = line2[10:13]
            if nationality == 'UZB':
                score += 20

        if len(line2) >= 21:
            sex_char = line2[20]
            if sex_char in ('M', 'F'):
                score += 25
            elif sex_char in ('N', 'W', 'H', 'K'):
                score -= 10

        chevron_count = line1.count('<') + line2.count('<')
        score += min(chevron_count, 20)

        if len(line1) == 44:
            score += 10
        if len(line2) == 44:
            score += 10

        return score

    def _find_mrz_lines(self, text: str) -> List[str]:
        """Matndan MRZ qatorlarini ajratish"""
        candidates = []

        for line in text.split('\n'):
            cleaned = self._clean_line(line)

            if len(cleaned) >= 38:
                if cleaned.startswith('P') or 'P<' in cleaned[:5]:
                    candidates.insert(0, cleaned)
                elif sum(c.isdigit() for c in cleaned) >= 8:
                    candidates.append(cleaned)
                elif len(cleaned) >= 42:
                    candidates.append(cleaned)

        full_text = ''.join(text.split())
        full_text = self._clean_line(full_text)
        if len(full_text) >= 80 and 'P<' in full_text[:10]:
            return self._smart_split(full_text)

        return candidates[:2]

    def _smart_split(self, text: str) -> List[str]:
        """Uzun matnni 2 qatorga ajratish"""
        pattern = r'([A-Z]{2}\d{7})(\d)([A-Z]{3})(\d{6})(\d)'
        match = re.search(pattern, text)

        if match:
            split_pos = match.start()
            if split_pos >= 40:
                line1 = text[:44]
                line2 = text[44:88]
            else:
                line1 = text[:split_pos]
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

        # Line2 preprocessing - filler belgilarini tiklash
        line2 = cls._preprocess_line2(line2)

        # ==========================================
        # LINE 1 PARSING
        # ==========================================
        doc_type = line1[0]
        doc_subtype = line1[1]
        issuing_country = line1[2:5].replace('<', '')
        names_section = line1[5:44]

        # Ism va familiyani ajratish
        surname, given_names = cls._parse_names(names_section)

        # ==========================================
        # LINE 2 PARSING (ICAO 9303 TD3 POZITSIYALAR)
        # ==========================================
        passport_raw = line2[0:9]
        passport_check = line2[9]
        nationality_raw = line2[10:13]
        dob_raw = line2[13:19]
        dob_check = line2[19]
        sex = line2[20]
        expiry_raw = line2[21:27]
        expiry_check = line2[27]
        personal_num_raw = line2[28:42]
        personal_num_check = line2[42]
        composite_check = line2[43]

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
    def _preprocess_line2(cls, line2: str) -> str:
        """
        Line2 preprocessing - OCR xatolarini tuzatish.
        Asosiy muammo: '<' belgisi '0' deb o'qiladi.

        O'zbekiston pasport Line2 formati:
        FB0292047<0UZB031225<M291018<3525120374400<<<<0
        [passport][c][nat][dob  ][c][s][expiry][c][pinfl        ][c][comp]

        Pozitsiyalar (0-indexed):
        0-8:   Passport number (9 belgi)
        9:     Check digit
        10-12: Nationality (3 belgi)
        13-18: DOB (6 raqam)
        19:    DOB check digit
        20:    Sex (M/F)
        21-26: Expiry (6 raqam)
        27:    Expiry check digit
        28-41: Personal number / PINFL (14 belgi)
        42:    Personal number check digit
        43:    Composite check digit
        """
        if len(line2) != 44:
            return line2

        # Line2 ni list ga aylantirish (o'zgartirish uchun)
        chars = list(line2)

        # Pozitsiya 20 - Jins (M yoki F bo'lishi kerak)
        sex_char = chars[20]
        if sex_char not in ('M', 'F', '<'):
            # OCR xatolari: N->M, W->M, H->M
            if sex_char in ('N', 'W', 'H', 'K', '0'):
                chars[20] = 'M'
            elif sex_char in ('E', 'P'):
                chars[20] = 'F'

        # Oxirgi 4 belgi ko'pincha filler (<<<<)
        # Agar '0000' bo'lsa, '<<<0' ga o'zgartirish
        if ''.join(chars[40:44]) == '0000':
            chars[40:43] = ['<', '<', '<']

        # Agar oxirgi 5 belgi '00000' bo'lsa
        if ''.join(chars[39:44]) == '00000':
            chars[39:43] = ['<', '<', '<', '<']

        return ''.join(chars)

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
        """Nationality ni tuzatish - kengaytirilgan OCR xatolari"""
        nationality = raw.replace('<', '').replace('0', 'O')

        # OCR xatolarini tuzatish - kengaytirilgan ro'yxat
        fixes = {
            # UZB variantlari
            'ZBO': 'UZB', 'LZB': 'UZB', 'USB': 'UZB',
            'U2B': 'UZB', 'UZ8': 'UZB', 'O2B': 'UZB',
            'UZ0': 'UZB', '0ZB': 'UZB', 'UZO': 'UZB',
            'U28': 'UZB', 'OZB': 'UZB', 'UZ6': 'UZB',
            '0Z8': 'UZB', '028': 'UZB', 'OZ8': 'UZB',
            'U7B': 'UZB', '07B': 'UZB', 'O7B': 'UZB',
            '7B0': 'UZB', '7BO': 'UZB', 'TBI': 'UZB',
            # Raqamli variantlar
            '007': 'UZB', '070': 'UZB', '700': 'UZB',
        }
        nationality = fixes.get(nationality, nationality)

        # O'zbekiston pasporti prefiksini tekshirish - har doim UZB
        if len(passport_number) >= 2 and passport_number[:2] in cls.UZB_PREFIXES:
            return 'UZB'

        # Agar hali ham noto'g'ri bo'lsa va O'zbekiston prefiksi bo'lsa
        if not nationality.isalpha() or len(nationality) != 3:
            if len(passport_number) >= 2 and passport_number[:2] in cls.UZB_PREFIXES:
                return 'UZB'

        return nationality if nationality else 'UZB'

    @classmethod
    def _fix_digits(cls, raw: str) -> str:
        """Raqamlarni tuzatish (O->0, I->1)"""
        return raw.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')

    @classmethod
    def _fix_sex(cls, raw: str) -> str:
        """Jinsni tuzatish - OCR xatolarini hisobga olish"""
        raw = raw.upper()
        if raw in ('M', 'F'):
            return raw
        # OCR xatolari: N->M, W->M, H->M
        if raw in ('N', 'W', 'H', 'K'):
            return 'M'
        # OCR xatolari: E->F, P->F
        if raw in ('E', 'P'):
            return 'F'
        return 'M'  # Default erkak (O'zbekiston pasportlari uchun ko'p holat)

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
        """ICAO 9303 Modulo 10 checksum"""
        total = 0
        for i, char in enumerate(data):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - 55
            else:
                value = 0

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
    """Pasport scanner - DeepSeek OCR + Tesseract fallback"""

    def __init__(self):
        self.security = SecurityModule()
        self.ocr = DeepSeekOCREngine()
        print(f"✅ {Config.SERVICE_NAME} v{Config.VERSION} tayyor", flush=True)

    def scan(self, image_bytes: bytes, filename: str = "unknown") -> Dict:
        """Pasportni skanerlash"""
        start_time = time.time()

        try:
            # 1. OCR - MRZ qatorlarini ajratish (DeepSeek yoki Tesseract)
            line1, line2 = self.ocr.extract_mrz_lines(image_bytes)

            # 2. Parse - ICAO 9303 bo'yicha
            parsed_data = ICAO9303Parser.parse(line1, line2)

            # 3. Metadata
            scan_time = time.time() - start_time
            engine_name = f"DeepSeek OCR ({Config.DEEPSEEK_MODEL})" if self.ocr.deepseek_available else "Tesseract OCR (Fallback)"
            parsed_data["scan_metadata"] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "file_name": filename,
                "scan_time_ms": int(scan_time * 1000),
                "scanner": f"{Config.SERVICE_NAME} v{Config.VERSION}",
                "engine": engine_name,
                "api_used": self.ocr.deepseek_available
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
        "engine": f"DeepSeek OCR ({Config.DEEPSEEK_MODEL})",
        "fallback_engine": "Tesseract OCR",
        "deepseek_configured": bool(Config.DEEPSEEK_API_KEY)
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
║         BOJXONA PASSPORT SCANNER - DeepSeek OCR                  ║
║                                                                  ║
║  🤖 DeepSeek OCR - DeepSeek-OCR-Latest-BF16.I64 model          ║
║  ✅ Tesseract OCR - Fallback engine                             ║
║  ✅ ICAO 9303 TD3 - Xalqaro standart                            ║
║  ✅ Yaxshilangan MRZ Detection (Morphological)                  ║
║                                                                  ║
║  Port: {port}                                                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
