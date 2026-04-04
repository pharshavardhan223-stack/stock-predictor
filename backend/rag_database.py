"""
FILE: backend/rag_database.py
PURPOSE: RAG Database for user trust, predictions, conversations, and daily tips
"""

import sqlite3
import json
from datetime import datetime

class RAGDatabase:
    def __init__(self, db_path="data/rag_knowledge.db"):
        self.db_path = db_path
        self._init_database()
        self._init_memory_tables()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # User trust score table
        c.execute('''CREATE TABLE IF NOT EXISTS user_trust (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            trust_score REAL DEFAULT 50.0,
            total_predictions INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            accuracy REAL DEFAULT 0.0,
            risk_profile TEXT DEFAULT 'moderate',
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Prediction history for RAG retrieval
        c.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            stock_symbol TEXT,
            prediction_date TIMESTAMP,
            predicted_action TEXT,
            actual_action TEXT,
            predicted_price REAL,
            actual_price REAL,
            confidence REAL,
            was_correct BOOLEAN,
            accuracy_score REAL
        )''')
        
        # User conversation context
        c.execute('''CREATE TABLE IF NOT EXISTS conversation_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            session_id TEXT,
            query TEXT,
            response TEXT,
            intent TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # User feedback
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            query TEXT,
            response TEXT,
            rating INTEGER,
            helpful BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
    
    def _init_memory_tables(self):
        """Create additional tables for memory features"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # User preferences
        c.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            daily_tips_enabled BOOLEAN DEFAULT 1,
            market_summary_enabled BOOLEAN DEFAULT 1,
            notification_preference TEXT DEFAULT 'all',
            favorite_stocks TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Conversation sessions
        c.execute('''CREATE TABLE IF NOT EXISTS conversation_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            session_id TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )''')
        
        # User corrections
        c.execute('''CREATE TABLE IF NOT EXISTS user_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            original_query TEXT,
            incorrect_response TEXT,
            correct_response TEXT,
            corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Daily tips
        c.execute('''CREATE TABLE IF NOT EXISTS daily_tips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tip_text TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            used_count INTEGER DEFAULT 0
        )''')
        
        conn.commit()
        conn.close()
        
        # Insert default tips
        self._insert_default_tips()
    
    def _insert_default_tips(self):
        """Insert default daily tips into database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM daily_tips")
        count = c.fetchone()[0]
        
        if count == 0:
            tips = [
                ("Always use stop-loss orders to protect your capital. Never risk more than 2% on a single trade.", "Risk Management"),
                ("The trend is your friend. Trade in the direction of the prevailing trend.", "Trading Psychology"),
                ("RSI below 30 indicates oversold conditions - potential buying opportunity.", "Technical Analysis"),
                ("RSI above 70 indicates overbought conditions - potential selling opportunity.", "Technical Analysis"),
                ("MACD crossover above signal line is bullish; below is bearish.", "Technical Analysis"),
                ("Diversify across different sectors to reduce portfolio risk.", "Portfolio Management"),
                ("Keep a trading journal to track your wins and losses. Learn from mistakes.", "Trading Psychology"),
                ("Volume confirms price movement. High volume breakouts are more reliable.", "Technical Analysis"),
                ("Never average down on losing positions. Cut losses quickly.", "Risk Management"),
                ("Set realistic profit targets. Don't be greedy.", "Trading Psychology"),
                ("Check economic calendar before trading - news events cause volatility.", "Fundamental Analysis"),
                ("Use multiple timeframes for confirmation - daily, 4h, 1h charts.", "Technical Analysis"),
                ("Fear and greed drive markets. Stay disciplined.", "Trading Psychology"),
                ("Paper trade new strategies before using real money.", "Risk Management"),
                ("Review your trades weekly to identify patterns.", "Performance Analysis")
            ]
            
            for tip, category in tips:
                c.execute("INSERT INTO daily_tips (tip_text, category) VALUES (?, ?)", (tip, category))
            
            conn.commit()
        
        conn.close()
    
    def update_trust_score(self, username, prediction_accuracy):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT trust_score, total_predictions, correct_predictions FROM user_trust WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result:
            trust_score, total, correct = result
            total += 1
            if prediction_accuracy > 0.7:
                correct += 1
                trust_score = min(100, trust_score + (prediction_accuracy * 5))
            else:
                trust_score = max(0, trust_score - 10)
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            c.execute('''UPDATE user_trust 
                        SET trust_score = ?, total_predictions = ?, correct_predictions = ?, 
                            accuracy = ?, last_active = CURRENT_TIMESTAMP
                        WHERE username = ?''',
                     (trust_score, total, correct, accuracy, username))
        else:
            trust_score = 50 + (prediction_accuracy * 20)
            c.execute('''INSERT INTO user_trust (username, trust_score, total_predictions, correct_predictions, accuracy)
                        VALUES (?, ?, 1, ?, ?)''',
                     (username, trust_score, 1 if prediction_accuracy > 0.7 else 0, prediction_accuracy * 100))
        
        conn.commit()
        conn.close()
        return trust_score
    
    def get_user_context(self, username, limit=10):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT trust_score, accuracy, risk_profile FROM user_trust WHERE username = ?", (username,))
        trust_data = c.fetchone()
        
        c.execute('''SELECT stock_symbol, predicted_action, was_correct, confidence
                    FROM prediction_history 
                    WHERE username = ? 
                    ORDER BY prediction_date DESC 
                    LIMIT ?''', (username, limit))
        predictions = c.fetchall()
        
        c.execute('''SELECT query, response, intent 
                    FROM conversation_context 
                    WHERE username = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 5''', (username,))
        conversations = c.fetchall()
        
        conn.close()
        
        return {
            "trust_score": trust_data[0] if trust_data else 50,
            "accuracy": trust_data[1] if trust_data else 0,
            "risk_profile": trust_data[2] if trust_data else "moderate",
            "total_predictions": trust_data[1] if trust_data else 0,
            "recent_predictions": predictions,
            "recent_conversations": conversations
        }
    
    def add_prediction_record(self, username, stock, predicted_action, actual_action, 
                              predicted_price, actual_price, confidence, was_correct):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        accuracy_score = 1.0 if was_correct else 0.0
        
        c.execute('''INSERT INTO prediction_history 
                    (username, stock_symbol, prediction_date, predicted_action, actual_action, 
                     predicted_price, actual_price, confidence, was_correct, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (username, stock, datetime.now(), predicted_action, actual_action,
                  predicted_price, actual_price, confidence, was_correct, accuracy_score))
        
        self.update_trust_score(username, accuracy_score)
        
        conn.commit()
        conn.close()
    
    def add_conversation(self, username, session_id, query, response, intent):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO conversation_context 
                    (username, session_id, query, response, intent)
                    VALUES (?, ?, ?, ?, ?)''',
                 (username, session_id, query, response, intent))
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, username, query, response, rating, helpful):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO feedback (username, query, response, rating, helpful)
                    VALUES (?, ?, ?, ?, ?)''',
                 (username, query, response, rating, helpful))
        
        conn.commit()
        conn.close()
    
    def get_similar_predictions(self, stock_symbol, limit=5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT predicted_action, was_correct, confidence, accuracy_score
                    FROM prediction_history 
                    WHERE stock_symbol = ?
                    ORDER BY prediction_date DESC 
                    LIMIT ?''', (stock_symbol, limit))
        
        results = c.fetchall()
        conn.close()
        
        return [{"action": r[0], "correct": r[1], "confidence": r[2], "accuracy": r[3]} for r in results]
    
    # ============================================================
    # DAILY TIP METHODS
    # ============================================================
    
    def get_daily_tip(self):
        """Get a random daily tip from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT tip_text, category FROM daily_tips ORDER BY RANDOM() LIMIT 1")
        result = c.fetchone()
        conn.close()
        
        if result:
            return {"tip": result[0], "category": result[1]}
        return None
    
    # ============================================================
    # CONVERSATION METHODS
    # ============================================================
    
    def get_conversation_summary(self, username, limit=5):
        """Get summary of recent conversations"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('''SELECT query, response, intent, timestamp 
                        FROM conversation_context 
                        WHERE username = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?''', (username, limit))
            results = c.fetchall()
        except:
            results = []
        
        conn.close()
        return [{"query": r[0], "response": r[1], "intent": r[2], "time": r[3]} for r in results]
    
    def add_conversation_session(self, username, session_id):
        """Start a new conversation session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("INSERT INTO conversation_sessions (username, session_id, start_time, message_count) VALUES (?, ?, datetime('now'), 0)", (username, session_id))
        conn.commit()
        conn.close()
    
    def update_session_message_count(self, session_id):
        """Increment message count for a session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute("UPDATE conversation_sessions SET message_count = message_count + 1 WHERE session_id = ?", (session_id,))
            conn.commit()
        except:
            pass
        
        conn.close()
    
    # ============================================================
    # USER CORRECTION METHODS
    # ============================================================
    
    def add_user_correction(self, username, original_query, incorrect_response, correct_response):
        """Store user correction for learning"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO user_corrections (username, original_query, incorrect_response, correct_response) 
                    VALUES (?, ?, ?, ?)''',
                  (username, original_query, incorrect_response, correct_response))
        
        conn.commit()
        conn.close()
    
    # ============================================================
    # USER PREFERENCES METHODS
    # ============================================================
    
    def get_user_preferences(self, username):
        """Get user preferences"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT daily_tips_enabled, market_summary_enabled, notification_preference, favorite_stocks FROM user_preferences WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                "daily_tips_enabled": bool(result[0]),
                "market_summary_enabled": bool(result[1]),
                "notification_preference": result[2],
                "favorite_stocks": result[3].split(',') if result[3] else []
            }
        else:
            return self.create_user_preferences(username)
    
    def create_user_preferences(self, username):
        """Create default user preferences"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("INSERT OR IGNORE INTO user_preferences (username, daily_tips_enabled, market_summary_enabled, notification_preference) VALUES (?, 1, 1, 'all')", (username,))
        conn.commit()
        conn.close()
        
        return {
            "daily_tips_enabled": True,
            "market_summary_enabled": True,
            "notification_preference": "all",
            "favorite_stocks": []
        }
    
    def update_user_preferences(self, username, preferences):
        """Update user preferences"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''UPDATE user_preferences 
                    SET daily_tips_enabled = ?, market_summary_enabled = ?, notification_preference = ?, favorite_stocks = ?
                    WHERE username = ?''',
                  (preferences.get('daily_tips_enabled', 1),
                   preferences.get('market_summary_enabled', 1),
                   preferences.get('notification_preference', 'all'),
                   ','.join(preferences.get('favorite_stocks', [])),
                   username))
        
        conn.commit()
        conn.close()
def get_conversation_summary(self, username, limit=5):
    """Get summary of recent conversations"""
    conn = sqlite3.connect(self.db_path)
    c = conn.cursor()
    
    try:
        c.execute('''SELECT query, response, intent, timestamp 
                    FROM conversation_context 
                    WHERE username = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?''', (username, limit))
        results = c.fetchall()
    except:
        results = []
    
    conn.close()
    return [{"query": r[0], "response": r[1], "intent": r[2], "time": r[3]} for r in results]

def get_user_preferences(self, username):
    """Get user preferences"""
    conn = sqlite3.connect(self.db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        daily_tips_enabled BOOLEAN DEFAULT 1,
        market_summary_enabled BOOLEAN DEFAULT 1,
        notification_preference TEXT DEFAULT 'all',
        favorite_stocks TEXT
    )''')
    
    c.execute("SELECT daily_tips_enabled, market_summary_enabled, notification_preference, favorite_stocks FROM user_preferences WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            "daily_tips_enabled": bool(result[0]),
            "market_summary_enabled": bool(result[1]),
            "notification_preference": result[2],
            "favorite_stocks": result[3].split(',') if result[3] else []
        }
    else:
        return self.create_user_preferences(username)

def create_user_preferences(self, username):
    """Create default user preferences"""
    conn = sqlite3.connect(self.db_path)
    c = conn.cursor()
    
    c.execute("INSERT OR IGNORE INTO user_preferences (username, daily_tips_enabled, market_summary_enabled, notification_preference) VALUES (?, 1, 1, 'all')", (username,))
    conn.commit()
    conn.close()
    
    return {
        "daily_tips_enabled": True,
        "market_summary_enabled": True,
        "notification_preference": "all",
        "favorite_stocks": []
    }

def update_user_preferences(self, username, preferences):
    """Update user preferences"""
    conn = sqlite3.connect(self.db_path)
    c = conn.cursor()
    
    c.execute('''UPDATE user_preferences 
                SET daily_tips_enabled = ?, market_summary_enabled = ?, notification_preference = ?, favorite_stocks = ?
                WHERE username = ?''',
              (preferences.get('daily_tips_enabled', 1),
               preferences.get('market_summary_enabled', 1),
               preferences.get('notification_preference', 'all'),
               ','.join(preferences.get('favorite_stocks', [])),
               username))
    
    conn.commit()
    conn.close()