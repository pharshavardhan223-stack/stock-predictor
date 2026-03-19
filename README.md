# 📈 Stock AI — Intelligent Stock Analysis Platform

> A full-stack AI-powered stock market analysis web app built with Flask, Python, and modern frontend technologies.  
> **Developer:** P Harsha Vardhan | B.Tech CSE

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔮 **AI Predictions** | ML-based price prediction using Linear Regression + technical indicators |
| 📊 **Live Stock Chart** | Candlestick/Line/Area chart with 8 overlay indicators (EMA, Bollinger, VWAP, S/R, Pivots, etc.) |
| 🛠️ **Drawing Tools** | Horizontal lines, trendlines, rays, Fibonacci retracements, zones & text labels |
| 🤖 **AI Predict Modal** | RSI, MACD, Bollinger, ATR scoring engine with 4 tabs: Predict / Chart Analysis / Key Levels / Strategy |
| ⚖️ **Stock Comparison** | Side-by-side live comparison of 2 stocks with metrics table, price chart & AI signal |
| 📰 **News Feed** | Stock news with sentiment scoring |
| ⭐ **Watchlist** | Add/remove stocks with price alerts |
| 💼 **Portfolio Advisor** | AI-driven portfolio recommendations |
| 🕐 **Prediction History** | Full log of past predictions with PDF export |
| 👤 **User Auth** | Login/logout, session management, admin panel |
| 📧 **Email Alerts** | Watchlist price alert notifications (requires config) |

---

## 🗂️ Project Structure

```
STOCK-/
├── backend/
│   ├── app.py                     # Main Flask app (~2170 lines, 35+ routes)
│   ├── config.py                  # App configuration
│   ├── routes/
│   │   ├── auth.py                # Login/logout/register routes
│   │   ├── new_routes.py          # Additional routes
│   │   └── portfolio_routes.py    # Portfolio advisor routes
│   └── utils/
│       ├── db_handler.py          # SQLite database operations
│       ├── predictor.py           # ML prediction engine
│       ├── auto_train.py          # Auto model retraining
│       ├── email_alerts.py        # Email notification system
│       ├── sentiment.py           # News sentiment analysis
│       ├── data_handler.py        # Data utilities
│       ├── stock_api.py           # yFinance API wrapper
│       ├── live_stock.py          # Live stock data handler
│       └── train_model.py         # Model training scripts
│
├── backend/templates/
│   ├── base.html                  # Base layout template
│   ├── login.html                 # Login page
│   ├── dashboard.html             # Main dashboard
│   ├── live_stock.html            # Advanced live chart (2274 lines)
│   ├── compare.html               # Stock comparison page
│   ├── predictions.html           # Prediction results
│   ├── watchlist.html             # Watchlist manager
│   ├── news.html                  # News feed
│   ├── portfolio_advisor.html     # Portfolio advisor
│   ├── history.html               # Prediction history
│   ├── admin.html                 # Admin panel
│   ├── edit_profile.html          # Edit profile
│   └── profile.html               # View profile
│
├── data/
│   ├── users.db                   # User accounts (SQLite)
│   ├── stockai.db                 # App data (SQLite)
│   └── history.csv                # Prediction history log
│
├── static/
│   └── images/logo.png
│
├── models/
│   └── linear_model.pkl           # Trained ML model
│
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pharshavardhan223-stack/stock-predictor.git
cd stock-predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```
flask
pandas
numpy
scikit-learn
yfinance
reportlab
plotly
requests
flask-mail          # for email alerts (optional)
python-dotenv       # for environment variables
```

### 4. Environment Variables (Optional but Recommended)

Create a `.env` file in the root directory:

```env
SECRET_KEY=your_secret_key_here
MAIL_USERNAME=your_gmail@gmail.com
MAIL_PASSWORD=your_gmail_app_password
MAIL_DEFAULT_SENDER=your_gmail@gmail.com
```

Then add to the top of `app.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 5. Run the Application

```bash
python -m backend.app
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🔐 Default Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| User | `user` | `1234` |

> ⚠️ Change these credentials before deploying to production.

---

## 🌐 Routes / Pages

| URL | Page | Description |
|---|---|---|
| `/` | Redirect | Redirects to `/dashboard` |
| `/login` | Login | Authentication page |
| `/logout` | Logout | Ends session |
| `/dashboard` | Dashboard | Overview with market summary |
| `/stock` | Predict | Stock symbol input & predictions |
| `/predictions` | Results | AI prediction output |
| `/watchlist` | Watchlist | Manage tracked stocks |
| `/news` | News | Stock news & sentiment |
| `/compare` | Compare | Side-by-side stock comparison |
| `/portfolio-advisor` | Advisor | Portfolio recommendations |
| `/history` | History | Prediction log + PDF export |
| `/stock-analysis` | Advanced AI | Live chart with full analysis |
| `/profile` | Profile | View user profile |
| `/edit-profile` | Edit Profile | Update user info |
| `/admin` | Admin | Admin panel (admin only) |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/compare-full/<symbol>` | GET | Full stock data for comparison page |
| `/api/news` | GET | Fetch stock news (query: `?symbol=AAPL`) |
| `/api/watchlist` | GET/POST | Watchlist data |
| `/alerts/remove` | POST | Remove price alert |
| `/download-report/<id>` | GET | Download prediction PDF |

---

## 📊 Live Chart Features (`/stock-analysis`)

### Chart Modes
- **Candlestick** — OHLC candles
- **Line** — Close price line
- **Area** — Filled area chart

### Timeframes
- 1m, 5m, 15m (intraday) · 1D, 1W, 1M (swing) · 3M, 6M, 1Y (long-term)

### Overlays / Indicators (toggleable)
- EMA 50 · Bollinger Bands · VWAP
- Support & Resistance · Trend Channel
- Pivot Points · Volume Histogram · MA20

### Drawing Tools
| Tool | Description |
|---|---|
| Pan | Default drag/scroll navigation |
| H-Line | Horizontal price level |
| Trendline | Draggable diagonal with slope % |
| Ray | Extends infinitely to the right |
| Fibonacci | 7 retracement levels (0–1) |
| Zone | Shaded rectangle with Δ% label |
| Label | Custom text annotation |

### AI Predict Modal (4 Tabs)
1. **Predict** — Score 0–100, direction, confidence, entry/target/stop
2. **Chart Analysis** — Pattern detection, trend summary
3. **Key Levels** — Support, resistance, pivot points
4. **Strategy** — Risk/Reward calculator, Multi-Timeframe analysis, News sentiment

---

## ⚖️ Stock Comparison Page (`/compare`)

Compare any two stocks (US or Indian — use `.NS` / `.BO` suffix for NSE/BSE):

**Displayed data:**
- Live price & 1-day change
- 52-week high/low
- Market cap, volume, P/E ratio, beta
- Period return (1M / 3M / 6M / 1Y)
- AI signal (BUY / SELL / HOLD)
- RSI, MACD, MA7, MA14, Bollinger Bands
- Price history comparison chart
- Side-by-side metrics table with winner highlighting

**Backend route:** `GET /api/compare-full/<symbol>?period=1mo`

The API must return a JSON object with these fields:

```json
{
  "price": 182.50,
  "change_pct": -0.35,
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "market_cap_fmt": "$2.85T",
  "volume": "55.2M",
  "high52": 199.62,
  "low52": 164.08,
  "pe_ratio": 33.01,
  "beta": 1.12,
  "period_return": -0.35,
  "avg_price": 184.20,
  "volatility": 1.8,
  "rsi": 44.2,
  "macd": -0.52,
  "ma7": 183.1,
  "ma14": 185.4,
  "bb_upper": 192.3,
  "bb_lower": 174.1,
  "signal": "HOLD",
  "dates": ["2024-01-01", "..."],
  "closes": [182.5, 183.1, "..."]
}
```

> ⚠️ **Note:** If 52W High/Low, Market Cap, Volume, or AI Signal show **N/A**, it means the `/api/compare-full/<symbol>` route is not returning those fields. Check `app.py` and ensure `yfinance` `.info` fields are correctly mapped.

---

## 🧠 ML Model

- **Algorithm:** Linear Regression (scikit-learn)
- **Features:** Close prices, volume, RSI-14, MACD, Bollinger Bands, EMA, ATR
- **Model file:** `models/linear_model.pkl`
- **Retraining:** Automatic via `utils/auto_train.py`

---

## 📧 Email Alerts Setup

1. Enable 2FA on your Gmail account
2. Generate an **App Password** at: https://myaccount.google.com/apppasswords
3. Add to `.env`:

```env
MAIL_USERNAME=your@gmail.com
MAIL_PASSWORD=abcd efgh ijkl mnop
```

4. Install Flask-Mail:

```bash
pip install Flask-Mail
```

---

## 🗄️ Database

The app uses **SQLite** (no setup required). Two databases:

- `data/users.db` — User accounts, sessions, watchlist, alerts
- `data/stockai.db` — Prediction history, portfolio data

---

## 🐛 Known Issues & Fixes

| Issue | Fix |
|---|---|
| 52W High/Low showing N/A | Ensure `yf.Ticker(sym).info["fiftyTwoWeekHigh"]` is returned from API route |
| Market Cap / Volume N/A | Map `market_cap_fmt` and `volume` from `yfinance` info dict |
| AI Signal showing `—` | Implement signal logic in `/api/compare-full` or pass from predictor |
| Graph not loading on live chart | Fixed: removed duplicate `let lastAnalysis` + corrected DOM pane order |
| Email alerts not sending | Requires Flask-Mail install + Gmail App Password in config |

---

## 🛡️ Security Notes

- Change default credentials before any public/production deployment
- Store secrets in `.env`, never hardcode in source files
- Add `.env` to `.gitignore`
- Use HTTPS in production

---

## 📁 `.gitignore` Recommendations

```
venv/
__pycache__/
*.pyc
.env
data/users.db
data/stockai.db
models/*.pkl
static/uploads/
```

---

## 📜 License

This project is for educational purposes — B.Tech CSE final year project.  
© 2024 P Harsha Vardhan. All rights reserved.

---

## 👨‍💻 Developer

**P Harsha Vardhan**  
B.Tech — Computer Science & Engineering  
GitHub: [@pharshavardhan223-stack](https://github.com/pharshavardhan223-stack)