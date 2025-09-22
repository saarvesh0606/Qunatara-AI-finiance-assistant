# app.py
import os
import io
import re
import json
import time
import fitz                       # PyMuPDF
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
from alpha_vantage.timeseries import TimeSeries
from flask_session import Session

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)
CORS(app, supports_credentials=True)

# IMPORTANT: A secret key is required for Flask sessions to work.
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "a-strong-default-secret-key-for-dev")

# Configure Server-Side Sessions
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure APIs
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"[WARN] Gemini configure: {e}")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# -----------------------------------------------------------------------------
# LLM helper
# -----------------------------------------------------------------------------
def call_gemini_with_retry(prompt, document_text="", retries=3, delay=5):
    if not os.getenv("GEMINI_API_KEY"):
        return "LLM not configured; returning a minimal fallback answer."
    generation_config = genai.types.GenerationConfig(temperature=0.2)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    for i in range(retries):
        try:
            parts = [prompt, document_text] if document_text else [prompt]
            resp = model.generate_content(parts, generation_config=generation_config)
            return resp.text
        except Exception as e:
            if "429" in str(e): time.sleep(delay); delay *= 2
            else: raise
    raise Exception("Gemini call failed after retries.")

# -----------------------------------------------------------------------------
# File text extraction
# -----------------------------------------------------------------------------
def extract_text_from_file(file):
    filename = (file.filename or "").lower()
    data = file.read()
    file.seek(0)
    if filename.endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    if filename.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if filename.endswith(".csv"):
        try: return data.decode("utf-8")
        except UnicodeDecodeError: return data.decode("latin1")
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        return df.to_string()
    raise ValueError("Unsupported file type")

# -----------------------------------------------------------------------------
# All Helper functions
# -----------------------------------------------------------------------------
def _num(x, ndigits=2):
    try:
        if pd.isna(x): return None
        return round(float(x), ndigits)
    except Exception:
        return None

def get_stock_data_alpha_vantage(ticker):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
        if data is None or data.empty:
            raise ValueError(f"No data returned from API for ticker '{ticker}'. It may be an invalid symbol.")
        data = data.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','5. volume':'Volume'})
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        raise ValueError(f"API Error for {ticker}: {str(e)}. You may have exceeded the free API limit of 25 calls/day.")

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff(); gain = (d.where(d > 0, 0)).rolling(period).mean(); loss = (-d.where(d < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan)); rsi = 100 - (100 / (1 + rs)); return rsi.fillna(50)

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast); ema_slow = _ema(close, slow); macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal); hist = macd - macd_signal; return macd, macd_signal, hist

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); close_series = out["Close"]; out["SMA20"] = close_series.rolling(20).mean()
    out["SMA50"] = close_series.rolling(50).mean(); out["SMA200"] = close_series.rolling(200).mean()
    out["EMA20"] = _ema(close_series, 20); out["RSI14"] = _rsi(close_series, 14)
    macd, sig, hist = _macd(close_series); out["MACD"] = macd; out["MACDSignal"] = sig; out["MACDHist"] = hist
    return out

def _naive_forecast(close_series: pd.Series, horizon: int = 5):
    returns = close_series.pct_change().dropna()
    if returns.empty: return []
    mu = returns.tail(20).mean(); sigma = returns.tail(20).std(); last = float(close_series.iloc[-1])
    out, p = [], last
    for i in range(1, horizon + 1):
        p = p * (1 + mu)
        out.append({"day": i, "price": round(p, 2), "upper": round(p * (1 + 2 * sigma), 2) if not np.isnan(sigma) else round(p, 2), "lower": round(p * (1 - 2 * sigma), 2) if not np.isnan(sigma) else round(p, 2),})
    return out

# -----------------------------------------------------------------------------
# All Flask Routes
# -----------------------------------------------------------------------------
@app.route("/api-status", methods=["GET"])
def api_status():
    return jsonify({ "news_api": bool(NEWS_API_KEY), "gemini": bool(os.getenv("GEMINI_API_KEY")), "status": "connected" if NEWS_API_KEY else "degraded" })

@app.route("/parse-csv-companies", methods=["POST"])
def parse_csv_companies():
    if "file" not in request.files: return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    try:
        content = extract_text_from_file(f)
        if not isinstance(content, str):
             content = pd.read_excel(f).to_string()
        
        df = pd.read_csv(io.StringIO(content))
        col = next((c for c in df.columns if c.strip() in ["Company", "Ticker", "Symbol"]), None)
        if not col: return jsonify({"error": "Could not find a 'Company', 'Ticker', or 'Symbol' column."}), 400
        companies = df[col].dropna().astype(str).unique().tolist()
        return jsonify({"companies": companies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-corporate", methods=["POST"])
def analyze_corporate_document():
    if "file" not in request.files: return jsonify({"error": "No file part"}), 400
    f = request.files["file"]; selected_company = request.form.get("company")
    try:
        content = extract_text_from_file(f)
        doc_text = ""
        if not isinstance(content, str):
             doc_text = pd.read_excel(f).to_markdown(index=False)
        else:
             doc_text = content
        
        if selected_company and f.filename.lower().endswith(".csv"):
            if isinstance(content, str):
                df = pd.read_csv(io.StringIO(content))
                col = next((c for c in df.columns if c.strip() in ["Company", "Ticker", "Symbol"]), None)
                if col:
                    company_df = df[df[col].astype(str) == str(selected_company)]
                    doc_text = company_df.to_markdown(index=False)

        prompt = """
        Analyze the provided financial data. Act as a data extraction engine.
        Your response MUST be a single, raw JSON object and nothing else. Do not add any conversational text, explanations, or markdown fences like ```json.
        The JSON object must contain these keys: "summary", "kpis", "chartData", "risks", "opportunities", "insightCard".
        - "summary": A concise executive summary as a string.
        - "kpis": An object with keys "Total Revenue", "Net Income", "EPS", "Operating Margin". The value for each KPI must be a FORMATTED STRING that includes currency (like $) and scale (like M for millions, B for billions). For example: "$514B", "₹2.5M", "-$0.27", "15.2%". Use the string "N/A" only if a value is completely missing from the document.
        - "chartData": An object with "labels" (an array of years as strings) and "revenueData" (an array of numbers, without formatting).
        - "risks": A single string containing a Markdown bulleted list of the top 3 risks.
        - "opportunities": A single string containing a Markdown bulleted list of the top 3 opportunities.
        - "insightCard": An object with "title" and "explanation" strings.
        """
        text = call_gemini_with_retry(prompt, doc_text)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON object found in model response.")
        
        payload = json.loads(json_match.group(0))
        app.config["LAST_DOCUMENT_TEXT"] = doc_text
        return jsonify(payload)
    except Exception as e:
        print(f"[CRITICAL] /analyze-corporate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/live-news", methods=["POST"])
def live_news_analysis():
    if not NEWS_API_KEY: return jsonify({"error": "News API key not configured."}), 500
    data = request.get_json(silent=True) or {}; company = (data.get("company") or "").strip()
    if not company: return jsonify({"error": "Company name is required"}), 400
    try:
        url ="https://newsapi.org/v2/everything"
        params = {"q": f"\"{company}\"", "searchIn": "title,description", "language": "en", "sortBy": "publishedAt", "pageSize": 10, "apiKey": NEWS_API_KEY,}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        news_data = r.json()
        if news_data.get("status") != "ok": return jsonify({"error": f"NewsAPI Error: {news_data.get('message', 'Unknown error')}"}), 502
        articles = news_data.get("articles", [])
        if not articles: return jsonify({"sentiment": "Neutral", "summary": f"No recent news found for '{company}'.", "articles": []})
        
        headlines = [f"- {a['title']}" for a in articles if a.get("title")]
        prompt = """
        Analyze the sentiment of the following recent news headlines.
        Respond ONLY with a single, raw JSON object with keys: "sentiment" and "summary".
        - "sentiment" must be one of: "Positive", "Negative", or "Neutral".
        - "summary" must be a single sentence of 25 words or less.
        """
        text_response = call_gemini_with_retry(prompt, "\n".join(headlines))
        
        json_match = re.search(r"\{.*\}", text_response, re.DOTALL)
        if not json_match:
            result = {"sentiment": "Neutral", "summary": "Could not determine sentiment from news analysis."}
        else:
            try:
                result = json.loads(json_match.group(0))
                if "sentiment" not in result or "summary" not in result:
                    result = {"sentiment": "Neutral", "summary": "Analysis response was incomplete."}
            except json.JSONDecodeError:
                result = {"sentiment": "Neutral", "summary": "Could not parse the analysis response."}

        clean_articles = [{"title": a.get("title"), "url": a.get("url"), "source": (a.get("source") or {}).get("name"), "publishedAt": a.get("publishedAt"), "description": a.get("description"),} for a in articles]
        result["articles"] = clean_articles
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        print(f"[CRITICAL] /live-news RequestException: {e}")
        return jsonify({"error": "Failed to connect to the News API."}), 503
    except Exception as e:
        print(f"[CRITICAL] /live-news: {e}"); return jsonify({"error": str(e)}), 500
    
@app.route("/stock-analysis", methods=["POST"])
def stock_analysis():
    data = request.get_json(silent=True) or {}; ticker = (data.get("ticker") or "").upper().strip(); period = data.get("period", "3mo"); interval = data.get("interval", "1d"); question = (data.get("question") or "").strip()
    if not ticker: return jsonify({"error": "Ticker is required"}), 400
    try:
        if not ALPHA_VANTAGE_API_KEY: return jsonify({"error": "Alpha Vantage API key is not configured on the server."}), 500
        df = get_stock_data_alpha_vantage(ticker)
        if df.empty: return jsonify({"error": f"No data for ticker '{ticker}'"}), 404
        if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index(); df = _compute_indicators(df); close = df["Close"].dropna()
        if close.empty: return jsonify({"error": "No valid closing prices available."}), 404
        latest_close = float(close.iloc[-1]); prev_close = float(close.iloc[-2]) if len(close) > 1 and pd.notna(close.iloc[-2]) else None
        change_pct = round(((latest_close / prev_close) - 1) * 100, 2) if (prev_close is not None and prev_close != 0.0) else 0.0
        def _last(col, nd=2): return _num(df[col].iloc[-1], nd) if (col in df.columns and pd.notna(df[col].iloc[-1])) else None
        metrics = {"price":_num(latest_close,2),"changePct":change_pct,"SMA20":_last("SMA20"),"SMA50":_last("SMA50"),"SMA200":_last("SMA200"),"EMA20":_last("EMA20"),"RSI14":_last("RSI14"),"MACD":_last("MACD",4),"MACDSignal":_last("MACDSignal",4),"MACDHist":_last("MACDHist",4)}
        forecast = _naive_forecast(close, horizon=5); as_of = df.index[-1].isoformat(); qa = None
        if question:
            context = f"Ticker: {ticker}\nAs of: {as_of}\nLatest Metrics:..."
            prompt = f'You are an equity analyst... question: "{question}"'
            qa = call_gemini_with_retry(prompt, context)
        hist_labels = [ts.isoformat() for ts in close.index]; hist_close = [round(float(v),2) for v in close.to_numpy()]
        return jsonify({"ticker":ticker,"asOf":as_of,"metrics":metrics,"forecast":forecast,"qa":qa,"history":{"labels":hist_labels,"close":hist_close}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/industry-analyze", methods=["POST"])
def industry_analyze():
    data = request.get_json(silent=True) or {}; industry = (data.get("industry") or "").strip(); inputs = (data.get("inputs") or "").strip()
    if not industry: return jsonify({"error": "Industry is required"}), 400
    try:
        prompt = f"""
        Act as a senior industry analyst providing a concise executive briefing on the "{industry}" sector.
        Consider these specific user inputs: {inputs if inputs else "None"}.
        
        Your response MUST be a single, raw JSON object with the following keys: "overview", "drivers", "risks", "outlook", "watchlist".
        - "overview": A professional, 3-4 sentence summary of the industry's current state.
        - "drivers": An array of 4 distinct strings, each detailing a key growth driver.
        - "risks": An array of 4 distinct strings, each detailing a key risk or headwind.
        - "outlook": A 2-3 sentence forward-looking statement for the next 6-12 months.
        - "watchlist": An array of 5 distinct strings, each a key metric or signal to monitor.
        """
        text = call_gemini_with_retry(prompt); m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m: raise ValueError("No valid JSON from model.")
        return jsonify(json.loads(m.group(0)))
    except Exception as e:
        print(f"[ERROR] /industry-analyze: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        style, text = "short", ""
        if "multipart/form-data" in request.content_type:
            if "file" not in request.files: return jsonify({"error": "No file part"}), 400
            f = request.files["file"]; text = extract_text_from_file(f); style = request.form.get("style", "short")
        else:
            data = request.get_json(silent=True) or {}; text = (data.get("text") or "").strip(); style = data.get("style", "short")
        if not text: return jsonify({"error": "No content provided."}), 400
        styles = {"short": "5-7 bullet points", "medium": "a tight paragraph", "long": "2 concise paragraphs"}
        prompt = f"Summarize the following content. Produce {styles.get(style, styles['short'])}."
        summary = call_gemini_with_retry(prompt, text); return jsonify({"summary": summary})
    except Exception as e:
        print(f"[ERROR] /summarize: {e}"); return jsonify({"error": str(e)}), 500

@app.route("/what-if-sandbox", methods=["POST"])
def what_if_sandbox():
    try:
        data = request.get_json(silent=True) or {}; query = (data.get("query") or "").strip()
        if not query: return jsonify({"error": "Query is required"}), 400
        context = app.config.get("LAST_DOCUMENT_TEXT")
        if not context: return jsonify({"error": "Please analyze a document first."}), 400
        prompt = f'Given the context of the previously analyzed financial data, answer this "what-if" in Markdown: "{query}"'
        text = call_gemini_with_retry(prompt, context); return jsonify({"answer": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-market", methods=["POST"])
def analyze_market_intelligence():
    try:
        files = request.files.getlist("files");
        if not files: return jsonify({"error": "No files provided"}), 400
        combined, names = "", []
        for f in files:
            try:
                t = extract_text_from_file(f); combined += f"\n\n--- Content from {f.filename} ---\n\n{t}"; names.append(f.filename)
            except Exception as e:
                print(f"[WARN] Could not process {f.filename}: {e}")
        if not combined.strip(): return jsonify({"error": "Could not extract text from files."}), 400
        prompt = f"As a market intelligence analyst, synthesize these documents... Sources: {', '.join(names)}"
        text = call_gemini_with_retry(prompt, combined); return jsonify({"briefing": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-advice", methods=["POST"])
def generate_customer_advice():
    try:
        data = request.get_json(silent=True) or {}; query, persona = (data.get("query") or "").strip(), (data.get("persona") or "").strip()
        if not query or not persona: return jsonify({"error": "Query and persona are required"}), 400
        prompt = f"As an AI financial assistant with the persona of a **{persona}**, respond to the user's query... User Query: \"{query}\""
        text = call_gemini_with_retry(prompt); return jsonify({"advice": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# Route to clear the file context from the session
# -----------------------------------------------------------------------------
@app.route("/clear_file_context", methods=["POST"])
def clear_file_context():
    if 'file_context' in session:
        session.pop('file_context')
        print("File context cleared from session.")
    return jsonify({"status": "cleared"})

# -----------------------------------------------------------------------------
# FINAL: AI Strategist Chatbot Route
# -----------------------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "").strip()
    chat_history_str = request.form.get("history", "[]")
    chat_history = json.loads(chat_history_str)
    uploaded_file = request.files.get("file")

    if not user_message and not uploaded_file:
        return jsonify({"error": "Empty message and no file received."}), 400

    file_context_for_prompt = ""
    persona_prompt = ""

    # --- File Handling & Persona Switching ---
    if uploaded_file or 'file_context' in session:
        persona_prompt = """
        You are 'QuanTara', a precise and diligent Data Analyst.
        Your SOLE task is to answer the user's questions based EXCLUSIVELY on the content of the document provided below.
        - Be direct and factual.
        - If the answer is not in the document, you MUST state: "The answer is not available in the provided document."
        - Do not use your general knowledge.
        """
        
        if uploaded_file:
            try:
                filename = uploaded_file.filename
                file_content = extract_text_from_file(uploaded_file)
                session['file_context'] = {"filename": filename, "content": file_content}
                print(f"File '{filename}' processed and stored in session.")
            except Exception as e:
                return jsonify({"error": f"Failed to process the uploaded file: {e}"}), 500

        if 'file_context' in session:
            context = session['file_context']
            file_context_for_prompt = context['content']
            
    else:
        # Persona for Investment Chatbot
        if 'profile' not in session:
            session['profile'] = {"goal": "Not set", "risk": "Not set"}
        user_profile = session['profile']
        
        persona_prompt = f"""
        You are 'QuanTara', a proactive and insightful AI Investment Strategist.
        **Your Guiding Principles:**
        1.  **Be Proactive:** Use the user's profile to provide a specific, direct answer immediately.
        2.  **Minimize Questions:** Only ask for clarification if a request is ambiguous AND the user's profile is not set.
        **User's Current Profile:**
        - Investment Goal: {user_profile['goal']}
        - Risk Tolerance: {user_profile['risk']}
        **Your Action Workflow:**
        - If the profile is NOT SET, ask for their goal and risk tolerance.
        - If the profile IS SET and they ask a general question, immediately suggest stocks that fit their profile.
        **PROFILE UPDATE INSTRUCTION:** If your response involves updating the user's profile, end your response with: """

    # --- Prompt Assembly ---
    formatted_history = "\n".join([f"  {msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    if user_message:
        formatted_history += f"\n  User: {user_message}"

    master_prompt = f"{persona_prompt}\n\nConversation History:\n{formatted_history}\n\nAssistant:"
    
    try:
        ai_response_text = call_gemini_with_retry(master_prompt, document_text=file_context_for_prompt)

        if 'file_context' not in session:
            profile_update_match = re.search(r'', ai_response_text)
            if profile_update_match:
                try:
                    new_profile_json = profile_update_match.group(1)
                    new_profile = json.loads(new_profile_json)
                    session['profile'] = new_profile
                    ai_response_text = ai_response_text.replace(profile_update_match.group(0), "").strip()
                    print(f"Updated session profile to: {session['profile']}")
                except Exception as e:
                    print(f"Error parsing profile update: {e}")

        return jsonify({"reply": ai_response_text})
    except Exception as e:
        print(f"[ERROR] /chat: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# NEW FEATURE 1: Holistic Financial Ecosystem Optimizer
# -----------------------------------------------------------------------------
@app.route("/optimize-finances", methods=["POST"])
def optimize_finances():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No financial data provided."}), 400

    session['financial_data'] = data

    prompt = f"""
    You are a holistic financial advisor AI. Your user has provided their complete financial picture.
    Financial Data: {json.dumps(data, indent=2)}

    Your task is to identify the single most impactful, actionable recommendation the user can take right now to improve their financial health.
    Analyze synergies between accounts. For example, is high-interest debt serviceable from a low-yield savings account?

    Provide your response as a single, raw JSON object with two keys: "recommendation" and "reasoning".
    - "recommendation": A short, direct call to action.
    - "reasoning": A concise, 2-3 sentence explanation of why this action is the most impactful.
    """
    try:
        response_text = call_gemini_with_retry(prompt)
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("The AI response did not contain a valid JSON object.")
        
        return jsonify(json.loads(json_match.group(0)))

    except Exception as e:
        print(f"[ERROR] /optimize-finances: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# NEW FEATURE 2: Generative Financial "Digital Twin"
# -----------------------------------------------------------------------------
@app.route("/digital-twin-query", methods=["POST"])
def digital_twin_query():
    data = request.get_json()
    user_query = data.get("query", "").strip()
    
    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    financial_data = session.get("financial_data")
    if not financial_data:
        return jsonify({"error": "No financial data found. Please use the Optimizer first to set your profile."}), 400

    prompt = f"""
    You are a financial modeling engine acting as the user's "Digital Twin".
    The user's current financial state is: {json.dumps(financial_data, indent=2)}
    The user's new goal is: "{user_query}"

    Your task is to generate a step-by-step, data-backed plan to help the user achieve their goal.
    Model the most probable outcomes based on their real financial data.
    
    Provide your response as a single, raw JSON object with one key: "plan".
    - "plan": A string containing a clear, step-by-step financial plan formatted in Markdown.
    """
    try:
        response_text = call_gemini_with_retry(prompt)
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("The AI response did not contain a valid JSON object.")

        return jsonify(json.loads(json_match.group(0)))

    except Exception as e:
        print(f"[ERROR] /digital-twin-query: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# NEW FEATURE 3: Proactive Financial Impact Analysis
# -----------------------------------------------------------------------------
@app.route("/proactive-impact-analysis", methods=["POST"])
def proactive_impact_analysis():
    data = request.get_json()
    portfolio = data.get("portfolio") 
    news_event = data.get("event", "").strip()

    if not portfolio or not news_event:
        return jsonify({"error": "A portfolio and a news event are required."}), 400

    prompt = f"""
    You are a financial contagion and ripple-effect analyst.
    A major market event has just occurred: "{news_event}"
    Your user holds the following stocks in their portfolio: {", ".join(portfolio)}

    Your task is to analyze the event and determine if it has a direct or indirect impact on any of the stocks in the user's portfolio.
    
    Provide your response as a single, raw JSON object with one key: "briefing".
    - "briefing": If there is a significant potential impact, provide a concise, one-paragraph briefing explaining the connection and the potential risk or opportunity. If there is no reasonably direct impact, the value should be the string "No significant impact identified for your portfolio."
    """
    try:
        response_text = call_gemini_with_retry(prompt)
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("The AI response did not contain a valid JSON object.")

        return jsonify(json.loads(json_match.group(0)))

    except Exception as e:
        print(f"[ERROR] /proactive-impact-analysis: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)