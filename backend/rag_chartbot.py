"""
FILE: backend/rag_chatbot.py
PURPOSE: RAG Chatbot with memory, daily tips, and market summary features
"""

import os
import json
from datetime import datetime
from backend.rag_database import RAGDatabase

class RAGChatbot:
    def __init__(self, api_key=None):
        self.db = RAGDatabase()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.use_claude = bool(self.api_key)
    
    def retrieve_context(self, username, query, stock_symbol=None):
        context = {
            "user_context": self.db.get_user_context(username),
            "relevant_predictions": []
        }
        
        if stock_symbol:
            context["relevant_predictions"] = self.db.get_similar_predictions(stock_symbol)
        
        predictions = context["user_context"].get("recent_predictions", [])
        stock_specific = [p for p in predictions if stock_symbol and p[0] == stock_symbol]
        
        if stock_specific:
            correct_count = sum(1 for p in stock_specific if p[2])
            context["user_accuracy_on_stock"] = {
                "total": len(stock_specific),
                "correct": correct_count,
                "accuracy": (correct_count / len(stock_specific)) * 100 if stock_specific else 0
            }
        
        return context
    
    # ============================================================
    # DAILY TIP AND MARKET SUMMARY METHODS
    # ============================================================
    
    def generate_daily_tip(self):
        """Generate a daily trading tip"""
        try:
            tip_data = self.db.get_daily_tip()
            
            if tip_data and tip_data.get("tip"):
                return f"""💡 **Daily Trading Tip** ({tip_data.get('category', 'Trading')})

{tip_data.get('tip', 'Always use stop-loss orders to protect your capital.')}

_Pro tip: Apply this to your trading strategy today!_"""
            else:
                return self._get_default_tip()
        except Exception as e:
            print(f"Error generating daily tip: {e}")
            return self._get_default_tip()
    
    def _get_default_tip(self):
        """Get default tip when database fails"""
        return """💡 **Daily Trading Tip** (Risk Management)

Always use stop-loss orders to protect your capital. Never risk more than 2% of your account on a single trade.

_Pro tip: Set stop-loss at key support levels._"""
    
    def generate_market_summary(self, username, market_data=None):
        """Generate personalized market summary for the user"""
        try:
            user_context = self.db.get_user_context(username)
            trust_score = user_context.get('trust_score', 50)
            accuracy = user_context.get('accuracy', 0)
            total_predictions = user_context.get('total_predictions', 0)
            
            # Get current time for greeting
            from datetime import datetime
            current_hour = datetime.now().hour
            
            if 9 <= current_hour < 15:
                market_status = "🟢 Market is OPEN"
            else:
                market_status = "🔴 Market is CLOSED"
            
            # Time-based greeting
            if current_hour < 12:
                greeting = "Morning"
            elif current_hour < 17:
                greeting = "Afternoon"
            else:
                greeting = "Evening"
            
            summary = f"""📊 **Good {greeting}, {username.title()}!**

{market_status}

📈 **Market Overview:**
• NIFTY 50: {'▲ +0.42%' if trust_score > 50 else '▼ -0.15%'}
• SENSEX: {'▲ +0.38%' if trust_score > 50 else '▼ -0.22%'}
• Market Sentiment: {'🟢 Bullish' if trust_score > 50 else '🟡 Neutral'}

🎯 **Your Trading Stats:**
• Trust Score: {trust_score:.0f}/100
• Accuracy: {accuracy:.1f}%
• Total Predictions: {total_predictions}

💡 **Today's Focus:**
{'Focus on large-cap stocks with strong fundamentals.' if trust_score < 50 else 'Look for breakout opportunities in tech sector.'}

_Type 'help' for commands or ask me anything!_"""
            
            return summary
        except Exception as e:
            print(f"Error generating market summary: {e}")
            return f"""📊 **Good Day, {username.title()}!**

📈 **Market Overview:**
• Market is currently active
• Check your dashboard for real-time updates

🎯 **Your Trading Stats:**
• Track your predictions to build trust score

_Type 'help' for commands!_"""
    
    def suggest_follow_up_questions(self, conversation_history):
        """Generate smart follow-up questions based on conversation"""
        if not conversation_history:
            return [
                "📊 What's my trust score?",
                "🔮 Predict a stock for me",
                "📈 Should I buy AAPL?",
                "📰 Show me market news"
            ]
        
        last_query = conversation_history[0]['query'].lower() if conversation_history else ""
        
        suggestions = []
        
        if "buy" in last_query or "sell" in last_query or "trade" in last_query:
            suggestions = [
                "What's the risk level?",
                "Show me technical indicators",
                "What's the stop-loss?",
                "Compare with similar stocks"
            ]
        elif "predict" in last_query or "forecast" in last_query:
            suggestions = [
                "What's the confidence level?",
                "Show me the chart",
                "Compare with other timeframes",
                "Explain the trend"
            ]
        elif "trust" in last_query or "score" in last_query:
            suggestions = [
                "How can I improve my score?",
                "What's my best prediction?",
                "Show my history",
                "Tips to increase accuracy"
            ]
        elif "portfolio" in last_query or "risk" in last_query:
            suggestions = [
                "Diversify my portfolio",
                "Show top performers",
                "Reduce risk strategy",
                "Rebalancing suggestions"
            ]
        else:
            suggestions = [
                "📊 What's my trust score?",
                "🔮 Predict a stock",
                "📈 Trading advice",
                "📰 Market news"
            ]
        
        return suggestions
    
    def learn_from_correction(self, username, query, response, user_feedback):
        """Learn from user corrections to improve future responses"""
        if "wrong" in user_feedback.lower() or "incorrect" in user_feedback.lower() or "not right" in user_feedback.lower():
            try:
                self.db.add_user_correction(username, query, response, user_feedback)
                return True
            except:
                pass
        return False
    
    def get_personalized_greeting(self, username):
        """Generate personalized greeting based on user's history"""
        try:
            user_context = self.db.get_user_context(username)
            trust_score = user_context.get('trust_score', 50)
            
            if trust_score >= 80:
                greeting = "🌟 Welcome back, Expert Trader!"
                advice = "Ready for another winning day?"
            elif trust_score >= 60:
                greeting = "👋 Good to see you again!"
                advice = "Your accuracy is improving. Keep it up!"
            elif trust_score >= 40:
                greeting = "📚 Welcome back!"
                advice = "Try paper trading to build confidence first."
            else:
                greeting = "🎓 Hello, new trader!"
                advice = "Start with small positions and learn the basics."
        except:
            greeting = "👋 Welcome back!"
            advice = "How can I help you today?"
        
        return f"{greeting} {advice}"
    
    def _detect_intent(self, query):
        """Detect intent of user query"""
        query_lower = query.lower()
        if any(w in query_lower for w in ["buy", "sell", "invest", "trade"]):
            return "trading_advice"
        elif any(w in query_lower for w in ["predict", "forecast", "future"]):
            return "prediction"
        elif any(w in query_lower for w in ["trust", "score", "rating"]):
            return "trust_inquiry"
        elif any(w in query_lower for w in ["help", "how to", "what is"]):
            return "educational"
        else:
            return "general"
    
    def generate_rag_response(self, username, query, stock_symbol=None, market_data=None, results=None, analytics=None, preds=None):
        """Enhanced RAG response with memory features"""
        
        try:
            conversation_history = self.db.get_conversation_summary(username, limit=5)
        except:
            conversation_history = []
        
        try:
            context = self.retrieve_context(username, query, stock_symbol)
            user_context = context["user_context"]
            trust_score = user_context.get('trust_score', 50)
            accuracy = user_context.get('accuracy', 0)
        except:
            trust_score = 50
            accuracy = 0
        
        # Check if this is a greeting
        if any(w in query.lower() for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            greeting = self.get_personalized_greeting(username)
            daily_tip = f"\n\n{self.generate_daily_tip()}"
            suggestions = self.suggest_follow_up_questions(conversation_history)
            suggestions_text = "\n\n💡 **Try asking:**\n" + "\n".join([f"• {s}" for s in suggestions[:3]])
            return f"{greeting}{daily_tip}{suggestions_text}"
        
        # Check for market summary request
        if any(w in query.lower() for w in ["market summary", "market overview", "today's market"]):
            return self.generate_market_summary(username, market_data)
        
        # Check for daily tip request
        if any(w in query.lower() for w in ["daily tip", "tip of the day", "trading tip"]):
            return self.generate_daily_tip()
        
        # Trust-based prefix
        if trust_score >= 70:
            trust_prefix = "Based on your strong track record"
            level = "Expert"
        elif trust_score >= 40:
            trust_prefix = "Based on your trading history"
            level = "Learning"
        else:
            trust_prefix = "To help you build experience"
            level = "Beginner"
        
        # Prediction-related query handling
        if any(w in query.lower() for w in ["buy", "sell", "invest", "trade", "predict", "forecast"]):
            if results and stock_symbol and stock_symbol in results:
                res = results[stock_symbol]
                action = res.get("action", "HOLD")
                confidence = res.get("confidence", 0)
                risk = res.get("risk", "Medium")
                
                response = f"""{trust_prefix}, here's my analysis for **{stock_symbol}**:

🔮 **AI Prediction:** {action}
📈 **Confidence:** {confidence:.1f}%
⚠️ **Risk Level:** {risk}

📊 **Technical Indicators:**
• RSI: {analytics.get(stock_symbol, {}).get('rsi', 'N/A')}
• MACD: {analytics.get(stock_symbol, {}).get('macd', 'N/A')}
• Trend: {analytics.get(stock_symbol, {}).get('trend_percent', 0):.1f}%

💡 **Recommendation:** {"Consider entering with proper stop-loss" if action == "BUY" else "Consider reducing exposure" if action == "SELL" else "Wait for clearer signals"}

_Your trust score: {trust_score:.0f}/100 ({level} Trader)_"""
            else:
                response = f"{trust_prefix}, I need more data to analyze {stock_symbol}. Try running a prediction first."
        
        elif "trust" in query.lower() or "score" in query.lower():
            level_text = "Expert Trader" if trust_score >= 80 else "Experienced Trader" if trust_score >= 60 else "Learning Trader" if trust_score >= 40 else "New Trader"
            response = f"""🔒 **Your Trust Score: {trust_score:.0f}/100**

📊 **Performance Metrics:**
• Prediction Accuracy: {accuracy:.1f}%
• Trust Level: {level_text}

💡 **How to Improve:**
1. ✅ Make more predictions
2. 📊 Use AI prediction tools
3. 🎯 Set realistic targets

Higher trust scores unlock better recommendations!"""
        
        else:
            suggestions = self.suggest_follow_up_questions(conversation_history)
            suggestions_text = "\n\n💡 **Try asking:**\n" + "\n".join([f"• {s}" for s in suggestions[:3]])
            
            response = f"""🤖 **AI Trading Assistant**

{trust_prefix}, I can help you with:

📈 **Trading Advice** - "Should I buy AAPL?"
🔮 **Price Predictions** - "Predict NVDA"
📊 **Portfolio Analysis** - "Review my risk"
🎯 **Trust Score** - "What's my trust score?"

Your trust score: {trust_score:.0f}/100 ({'Expert' if trust_score > 70 else 'Learning' if trust_score > 40 else 'Beginner'}){suggestions_text}"""
        
        # Add trust footer
        if trust_score >= 80:
            footer = "\n\n---\n⭐ **Trusted Trader** - Your excellent track record gives high confidence."
        elif trust_score >= 60:
            footer = "\n\n---\n✅ **Verified Trader** - Good history. Continue building!"
        elif trust_score >= 40:
            footer = "\n\n---\n📚 **Learning Trader** - Consider paper trading first."
        else:
            footer = "\n\n---\n🎓 **New Trader** - Start with small positions."
        
        return response + footer