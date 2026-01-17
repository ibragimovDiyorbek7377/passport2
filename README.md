# 🛂 Bojxona Passport Scanner

**Customs-Grade ICAO 9303 TD3 Compliant Passport Scanner for Uzbekistan**

A professional-grade Telegram Mini App for scanning and validating Uzbekistan Biometric Passports with strict ICAO 9303 TD3 standard compliance.

---

## 🎯 Overview

This system is designed for the **Customs Committee of Uzbekistan** to provide accurate, real-time passport scanning and validation through a Telegram Mini App interface.

### Key Features

- ✅ **ICAO 9303 TD3 Standard Compliance** - Full MRZ parsing with checksum validation
- ✅ **JSHSHIR/PNFL Extraction** - Uzbekistan Personal Number (14 digits) extraction
- ✅ **Real-time Camera Scanning** - Mobile-optimized camera interface
- ✅ **Professional UI** - Dark mode with `#00F721` neon green branding
- ✅ **Automatic Validation** - Modulus 10 checksum verification
- ✅ **Railway.app Optimized** - Docker-based deployment with Tesseract OCR

---

## 🏗️ Architecture

### Technology Stack

**Frontend:**
- HTML5 + CSS3 + Vanilla JavaScript
- Telegram Web App SDK
- Camera API for mobile scanning

**Backend:**
- Python 3.9
- FastAPI (REST API)
- PassportEye (OCR Engine based on Tesseract)
- OpenCV (Image preprocessing)
- python-telegram-bot (Bot integration)

**Deployment:**
- Railway.app
- Docker containerization
- Tesseract OCR 4.x

---

## 📋 System Requirements

### Extracted Data Fields

The system extracts and validates the following fields from Uzbekistan Biometric Passports:

| Field | Description | Format |
|-------|-------------|--------|
| **Passport Number** | 9-character passport ID | `AA1234567` |
| **Surname** | Family name | Text |
| **Given Names** | First and middle names | Text |
| **Date of Birth** | Birth date | `DD.MM.YYYY` |
| **Sex** | Gender | `M` / `F` |
| **Date of Expiry** | Expiration date | `DD.MM.YYYY` |
| **Personal Number** | JSHSHIR/PNFL (14 digits) | `12345678901234` |
| **Nationality** | Country code | `UZB` |

### ICAO 9303 Validation

The system performs the following validations:

1. **Passport Number Checksum** - Modulus 10 with weights 7,3,1
2. **Date of Birth Checksum** - Validates DOB check digit
3. **Date of Expiry Checksum** - Validates expiry check digit
4. **Personal Number Checksum** - Validates JSHSHIR/PNFL
5. **Composite Checksum** - Overall MRZ integrity check
6. **Date Format Validation** - YYMMDD format verification

---

## 🚀 Deployment Guide

### Railway.app Deployment

1. **Connect Repository to Railway:**
   ```bash
   railway login
   railway link
   ```

2. **Environment Variables:**

   Set the following in Railway dashboard:
   ```env
   PORT=8000
   RAILWAY_STATIC_URL=https://your-app.railway.app
   ```

3. **Deploy:**
   ```bash
   railway up
   ```

Railway will automatically:
- Detect the `Dockerfile`
- Build with Tesseract OCR support
- Deploy on the specified port

### Manual Docker Deployment

```bash
# Build the image
docker build -t bojxona-scanner .

# Run the container
docker run -d -p 8000:8000 \
  -e PORT=8000 \
  bojxona-scanner
```

---

## 🔧 Configuration

### Telegram Bot Setup

1. **Create Bot:**
   - Message [@BotFather](https://t.me/botfather)
   - Create new bot: `/newbot`
   - Get bot token: `8319403923:AAH8LkaqsickGL980lJKEOk51tVyhI6onZA`

2. **Set Mini App URL:**
   ```
   /setmenubutton
   Select your bot
   Provide Mini App URL: https://your-app.railway.app
   ```

3. **Update bot.py:**
   - The bot token is already configured in `bot.py`
   - Update `WEBAPP_URL` environment variable

---

## 📱 Usage

### For Users

1. **Start Bot:**
   - Open Telegram
   - Search for your bot
   - Send `/start`

2. **Scan Passport:**
   - Click "📷 Pasportni Skanerlash"
   - Allow camera permissions
   - Align passport within the neon green frame
   - Click "Rasmga Olish" to capture
   - View results with validation status

3. **Available Commands:**
   - `/start` - Launch Mini App
   - `/help` - User guide
   - `/info` - System information

---

## 🎨 Design System

### Brand Colors

```css
--neon-green: #00F721  /* Primary brand color */
--dark-bg: #0a0a0a     /* Background */
--dark-card: #151515   /* Card background */
--text-primary: #ffffff
--text-secondary: #b0b0b0
```

### UI Elements

- **Scanner Frame:** `#00F721` with glow effect
- **Success State:** Animated glow with `#00F721`
- **Dark Mode:** Professional Bojxona (Customs) theme
- **Responsive:** Optimized for mobile devices

---

## 🔐 Security Considerations

### Data Privacy

- ❌ **No data storage** - All processing is done in real-time
- ✅ **Session-based** - Data is not persisted
- ✅ **Telegram encryption** - Uses Telegram's secure infrastructure
- ✅ **HTTPS only** - Enforced for production

### Validation Security

- Strict ICAO 9303 checksum validation
- Multi-level verification (passport, DOB, expiry, JSHSHIR, composite)
- Format validation for all date fields
- Detection of tampered or invalid MRZ data

---

## 📊 API Documentation

### Endpoints

#### `POST /scan`

Scan passport image and extract MRZ data.

**Request:**
```http
POST /scan HTTP/1.1
Content-Type: multipart/form-data

file: <passport_image.jpg>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "passport_number": "AA1234567",
    "surname": "KARIMOV",
    "given_names": "AZIZ",
    "date_of_birth": "15.03.1990",
    "sex": "M",
    "date_of_expiry": "15.03.2030",
    "personal_number": "12345678901234",
    "nationality": "UZB",
    "validation_status": "PASS",
    "validations": {
      "passport_number_valid": true,
      "dob_valid": true,
      "expiry_valid": true,
      "personal_number_valid": true,
      "composite_valid": true,
      "dob_format_valid": true,
      "expiry_format_valid": true
    }
  }
}
```

#### `GET /health`

Health check endpoint.

```json
{
  "status": "healthy",
  "service": "bojxona-passport-scanner"
}
```

#### `GET /test`

Test endpoint to verify API operational status.

---

## 🧪 Testing

### Manual Testing

1. **Test with Sample Passport:**
   ```bash
   curl -X POST http://localhost:8000/scan \
     -F "file=@sample_passport.jpg"
   ```

2. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

### Expected Results

- **Valid Passport:** `validation_status: "PASS"`
- **Invalid Checksum:** `validation_status: "FAIL"`
- **No MRZ Detected:** HTTP 422 error

---

## 📦 Project Structure

```
passporteye/
├── Dockerfile              # Railway-optimized Docker configuration
├── requirements.txt        # Python dependencies
├── railway.json            # Railway deployment config
├── main.py                 # FastAPI backend with ICAO validation
├── bot.py                  # Telegram bot integration
├── templates/
│   └── index.html         # TMA frontend with camera interface
├── static/                # Static assets (if needed)
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 🛠️ Development

### Local Setup

1. **Clone Repository:**
   ```bash
   git clone https://github.com/yourusername/passporteye.git
   cd passporteye
   ```

2. **Install Dependencies:**
   ```bash
   # Install Tesseract (Ubuntu/Debian)
   sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1

   # Install Python packages
   pip install -r requirements.txt
   ```

3. **Run Locally:**
   ```bash
   python bot.py
   ```

4. **Access:**
   - API: `http://localhost:8000`
   - Mini App: `http://localhost:8000`

### Environment Variables

```env
PORT=8000
RAILWAY_STATIC_URL=http://localhost:8000
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Tesseract not found
```bash
# Solution: Install Tesseract
apt-get install tesseract-ocr
```

**Issue:** MRZ not detected
- Ensure good lighting
- Align passport within guide frame
- Check image quality (minimum 1280x720)

**Issue:** Railway build fails
- Verify Dockerfile has Tesseract installation
- Check requirements.txt for correct versions

---

## 📜 License & Legal

### ICAO 9303 Compliance

This system adheres to:
- ICAO Doc 9303 Part 4 (MRZ specifications)
- TD3 format (two lines, 44 characters each)
- ISO/IEC 7501-1 standards

### Usage Rights

**For Customs Committee of Uzbekistan Use Only**

- ✅ Authorized for official customs operations
- ❌ Not for commercial redistribution
- ⚠️ Handle all scanned data per data protection regulations

---

## 👨‍💻 Development Team

**Lead Engineer:** Senior Computer Vision & GovTech Specialist

**Contact:**
- Technical Support: Via Telegram bot
- Issues: GitHub Issues
- Documentation: This README

---

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Initial release
- ✅ ICAO 9303 TD3 parsing
- ✅ JSHSHIR/PNFL extraction
- ✅ Telegram Mini App interface
- ✅ Railway.app deployment support
- ✅ Dark mode UI with #00F721 branding

---

## 🚧 Roadmap

### Planned Features
- [ ] Batch scanning support
- [ ] PDF report generation
- [ ] Admin dashboard
- [ ] Advanced analytics
- [ ] Multi-language support (Russian, English)
- [ ] ID card scanning (biometric ID cards)

---

## 📞 Support

For technical support or feature requests:

1. **Telegram:** Contact via the bot
2. **GitHub Issues:** Report bugs and request features
3. **Email:** [Support email if applicable]

---

**Built with ❤️ for the Customs Committee of Uzbekistan**

*Powered by PassportEye OCR | ICAO 9303 Compliant*
