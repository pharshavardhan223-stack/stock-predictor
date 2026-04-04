"""
FILE: backend/routes/advanced_analysis.py
PURPOSE: Advanced ML/Deep Learning analysis routes for uploaded datasets with RAG integration
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import traceback

# Fix path for reports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Import advanced models
import sys
sys.path.insert(0, BASE_DIR)
from backend.utils.advanced_models import AdvancedPredictor
from backend.utils.report_generator import ReportGenerator

# Create blueprint
advanced_analysis_bp = Blueprint("advanced_analysis", __name__, url_prefix="/advanced")

# Initialize predictors
predictor = AdvancedPredictor()
report_gen = ReportGenerator()


# ============================================================
# RAG ANALYSIS FUNCTIONS
# ============================================================

def generate_rag_analysis_response(question, context):
    """Generate RAG response for analysis questions"""
    
    column = context.get("column", "this data")
    trend = context.get("trend", "stable")
    trend_percent = context.get("trend_percent", 0)
    confidence = context.get("confidence", 0)
    recommendation = context.get("recommendation", "HOLD")
    current_value = context.get("current_value", 0)
    predicted_value = context.get("predicted_value", 0)
    has_anomalies = context.get("has_anomalies", False)
    anomaly_count = context.get("anomaly_count", 0)
    model_used = context.get("model_used", "AI Model")
    data_points = context.get("data_points", 0)
    r2_score = context.get("r2_score", 0)
    
    # Trend interpretation
    if trend == "up":
        trend_explanation = f"INCREASE by {trend_percent:.1f}%"
        trend_advice = "This suggests growth potential. Consider increasing exposure."
    elif trend == "down":
        trend_explanation = f"DECREASE by {trend_percent:.1f}%"
        trend_advice = "This suggests decline. Consider reducing exposure or using stop-losses."
    else:
        trend_explanation = "remain STABLE"
        trend_advice = "Wait for clearer signals before making moves."
    
    q = question.lower()
    
    # Answer based on question type
    if "trend" in q or "direction" in q:
        return f"""📈 **Trend Analysis**

The data for **{column}** shows a {trend_explanation}.

{trend_advice}

🔍 **Details:**
• Current Value: {current_value:,.2f}
• Predicted Value: {predicted_value:,.2f}
• Change: {trend_percent:.1f}%

_This analysis is based on {data_points} data points using {model_used} model._"""
    
    elif "confidence" in q or "accurate" in q or "reliable" in q:
        if confidence >= 80:
            confidence_text = "HIGH confidence"
            advice = "You can rely on this prediction with good certainty."
        elif confidence >= 60:
            confidence_text = "MODERATE confidence"
            advice = "Consider using alongside other indicators."
        else:
            confidence_text = "LOW confidence"
            advice = "Data may be volatile. Use caution and consider more data."
        
        return f"""🎯 **Confidence Analysis**

The model has **{confidence_text} ({confidence:.0f}%)** in this prediction.

📊 **Why?**
• R² Score: {r2_score:.4f} (1.0 = perfect fit)
• Data Points: {data_points}
• Model Used: {model_used}

💡 **Advice:** {advice}"""
    
    elif "anomaly" in q or "unusual" in q or "pattern" in q:
        if has_anomalies:
            return f"""⚠️ **Anomaly Detection**

Found **{anomaly_count} unusual pattern(s)** in your data.

🔍 **What this means:**
Anomalies are data points that deviate significantly from normal patterns. They could indicate:
• Data entry errors
• One-time events (sales, promotions)
• Market volatility spikes
• Seasonal effects

💡 **Recommendation:** Investigate these points. If they are valid, consider them in your analysis. If they are errors, consider removing them."""
        else:
            return f"""✅ **No Anomalies Detected**

Your data for **{column}** appears clean and consistent.

📊 **What this means:**
No unusual patterns were found. The data follows a normal pattern, which increases prediction reliability.

💡 **Advice:** You can trust the trend analysis for this dataset."""
    
    elif "recommend" in q or "action" in q or "should i" in q:
        if "buy" in q or "invest" in q:
            if trend == "up":
                action = "✅ CONSIDER BUYING/INCREASING"
                reason = f"Upward trend of {trend_percent:.1f}% suggests growth potential."
            elif trend == "down":
                action = "⚠️ AVOID or CONSIDER SELLING"
                reason = f"Downward trend of {trend_percent:.1f}% suggests potential decline."
            else:
                action = "➡️ HOLD POSITION"
                reason = "Stable trend suggests waiting for clearer signals."
            
            return f"""💡 **Investment Recommendation**

**Action:** {action}

**Reason:** {reason}

📊 **Supporting Data:**
• Confidence Level: {confidence:.0f}%
• Current Value: {current_value:,.2f}
• Predicted Value: {predicted_value:,.2f}

⚠️ *Always use proper risk management and stop-loss orders.*"""
        
        else:
            return f"""💡 **Recommendation Summary**

Based on the analysis of **{column}**:

**Action:** {recommendation}

**Confidence:** {confidence:.0f}%

**Key Insight:** The trend is {trend} with a {trend_percent:.1f}% expected change.

{trend_advice}"""
    
    elif "explain" in q or "what does" in q or "meaning" in q:
        return f"""📖 **Analysis Explanation**

**What was analyzed?** {column} - {data_points} data points

**What model was used?** {model_used} (automatically selected for best accuracy)

**What does the trend mean?** 
The data shows a {trend} trend, meaning values are expected to {trend_explanation.lower()}.

**How confident is the model?** 
{confidence:.0f}% confident (R² score: {r2_score:.4f})

**Any unusual patterns?** 
{'Yes' if has_anomalies else 'No'} - {'Investigate these points' if has_anomalies else 'Data appears clean'}

💡 **Plain English:** {trend_advice}"""
    
    elif "model" in q or "how it works" in q:
        return f"""🤖 **About the AI Model**

**Model Used:** {model_used}

**How it works:**
This model analyzes historical patterns in your {column} data to predict future values. It looks at trends, seasonality, and relationships in the data.

**Why this model?**
The system automatically selected {model_used} because it performed best on your data (R² score: {r2_score:.4f}).

**Limitations:**
• Predictions are based on historical patterns only
• Unexpected events can affect accuracy
• More data generally improves predictions

💡 **Tip:** Run analysis regularly to track how predictions compare with actual outcomes."""
    
    elif "help" in q or "what can i ask" in q:
        return f"""❓ **What You Can Ask Me**

Try these questions about your analysis:

📈 **Trend Questions:**
• "What is the trend direction?"
• "Is the trend strong or weak?"

🎯 **Confidence Questions:**
• "How confident is this prediction?"
• "Should I trust this analysis?"

⚠️ **Anomaly Questions:**
• "Are there any anomalies?"
• "What do anomalies mean?"

💡 **Recommendation Questions:**
• "Should I buy or sell?"
• "What action should I take?"

📖 **Explanation Questions:**
• "Explain this analysis in simple terms"
• "What does the R² score mean?"

🤖 **Model Questions:**
• "How does the AI model work?"
• "Why was this model selected?"

💬 **Just type your question naturally!**"""
    
    else:
        # Default response with summary
        return f"""📊 **Analysis Summary for {column}**

**Trend:** {trend.upper()} - Expected {trend_explanation}
**Confidence:** {confidence:.0f}%
**Recommendation:** {recommendation}

📈 **Key Numbers:**
• Current: {current_value:,.2f}
• Predicted: {predicted_value:,.2f}
• Change: {trend_percent:.1f}%

{'⚠️ ' + str(anomaly_count) + ' anomaly(ies) detected - ask about anomalies' if has_anomalies else '✅ No anomalies detected'}

💡 **What would you like to know?** Try asking:
• "Explain the trend"
• "How confident are you?"
• "What should I do?"
• "Tell me about anomalies" """


# ============================================================
# ROUTES
# ============================================================

@advanced_analysis_bp.route("/analysis", methods=["GET", "POST"])
def advanced_analysis():
    """Main analysis page - model selection and settings"""
    if "user" not in session:
        flash("Please login first", "error")
        return redirect("/login")
    
    file_path = session.get("advanced_file_path")
    columns = session.get("advanced_columns", [])
    selected_column = session.get("advanced_selected_column")
    
    if not file_path or not os.path.exists(file_path):
        flash("Please upload a file first", "error")
        return redirect("/upload")
    
    try:
        df = pd.read_csv(file_path)
        
        # Convert string columns to numeric where possible
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(r'[\$,₹]', '', regex=True)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        preview_data = df.head(10).to_html(classes="table table-striped", index=False)
        data_shape = df.shape
    except Exception as e:
        preview_data = f"<p>Error loading data: {e}</p>"
        data_shape = (0, 0)
    
    # Available models
    models = [
        {"id": "auto", "name": "Auto (Best Model)", "type": "AI", "description": "Automatically selects best performing model", "icon": "fa-magic"},
        {"id": "linear_regression", "name": "Linear Regression", "type": "ML", "description": "Best for linear trends", "icon": "fa-chart-line"},
        {"id": "random_forest", "name": "Random Forest", "type": "ML", "description": "Handles complex patterns", "icon": "fa-tree"},
        {"id": "gradient_boosting", "name": "Gradient Boosting", "type": "ML", "description": "High accuracy predictions", "icon": "fa-chart-simple"},
        {"id": "ensemble", "name": "Ensemble (Voting)", "type": "Combined", "description": "Uses all models + voting", "icon": "fa-cubes"}
    ]
    
    forecast_periods = [
        {"days": 7, "label": "7 Days (1 Week)"},
        {"days": 14, "label": "14 Days (2 Weeks)"},
        {"days": 30, "label": "30 Days (1 Month)"},
        {"days": 90, "label": "90 Days (3 Months)"}
    ]
    
    return render_template("advanced_analysis.html",
                         preview_data=preview_data,
                         data_shape=data_shape,
                         columns=columns,
                         selected_column=selected_column,
                         models=models,
                         forecast_periods=forecast_periods,
                         username=session.get("user"))


@advanced_analysis_bp.route("/run-prediction", methods=["POST"])
def run_advanced_prediction():
    """Run prediction with selected model and settings"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        model_id = request.form.get("model", "auto")
        forecast_days = int(request.form.get("forecast_days", 30))
        column = request.form.get("column", "")
        enable_anomaly = request.form.get("enable_anomaly") == "on"
        generate_insights = request.form.get("generate_insights") == "on"
        
        file_path = session.get("advanced_file_path")
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400
        
        if not column:
            return jsonify({"error": "Please select a column"}), 400
        
        df = pd.read_csv(file_path)
        
        # Convert string to numeric
        if column in df.columns and df[column].dtype == 'object':
            try:
                df[column] = df[column].astype(str).str.replace('[\$,₹]', '', regex=True)
                df[column] = df[column].str.replace(',', '', regex=False)
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except:
                pass
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found"}), 400
        
        series = df[column].dropna().values
        
        if len(series) < 5:
            return jsonify({"error": f"Need at least 5 valid numeric values. Found {len(series)}."}), 400
        
        # Auto model selection
        if model_id == "auto":
            model_id = predictor.find_best_model(series)
            print(f"Auto selected model: {model_id}")
        
        result = predictor.predict(
            series=series,
            model_type=model_id,
            forecast_days=forecast_days,
            column_name=column,
            enable_anomaly=enable_anomaly
        )
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        if generate_insights:
            insights = predictor.generate_insights(series, result, column)
            result["insights"] = insights
        
        # Store analysis context for RAG
        session["advanced_result"] = result
        session["advanced_column"] = column
        session["advanced_model"] = model_id
        session["advanced_forecast_days"] = forecast_days
        
        # Store detailed context for RAG questions
        session["advanced_analysis_context"] = {
            "column": column,
            "trend": result.get("trend", "stable"),
            "trend_percent": result.get("trend_percent", 0),
            "confidence": result.get("confidence", 0),
            "recommendation": result.get("recommendation", "HOLD"),
            "current_value": result.get("current_value", 0),
            "predicted_value": result.get("predicted_value", 0),
            "has_anomalies": len(result.get("anomalies", [])) > 0,
            "anomaly_count": len(result.get("anomalies", [])),
            "model_used": result.get("model_display_name", "AI Model"),
            "data_points": result.get("metrics", {}).get("data_points", 0),
            "r2_score": result.get("metrics", {}).get("r2", 0)
        }
        
        return jsonify({
            "success": True,
            "result": result,
            "redirect": "/advanced/report"
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@advanced_analysis_bp.route("/report")
def advanced_report():
    """Display the analysis report"""
    if "user" not in session:
        return redirect("/login")
    
    result = session.get("advanced_result")
    column = session.get("advanced_column")
    model = session.get("advanced_model")
    forecast_days = session.get("advanced_forecast_days")
    
    if not result:
        flash("No analysis result found. Please run prediction first.", "error")
        return redirect("/advanced/analysis")
    
    model_names = {
        "auto": "Auto (Best Model)",
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "ensemble": "Ensemble (Best Model)"
    }
    model_name = model_names.get(model, "AI Model")
    
    return render_template("advanced_report.html",
                         result=result,
                         column=column,
                         model_name=model_name,
                         forecast_days=forecast_days,
                         username=session.get("user"),
                         generated_date=datetime.now().strftime("%d %B %Y, %I:%M %p"))


@advanced_analysis_bp.route("/export-report")
def export_advanced_report():
    """Export report as PDF"""
    if "user" not in session:
        return redirect("/login")
    
    result = session.get("advanced_result")
    column = session.get("advanced_column")
    model = session.get("advanced_model")
    forecast_days = session.get("advanced_forecast_days")
    
    if not result:
        flash("No report to export", "error")
        return redirect("/advanced/analysis")
    
    # Generate PDF report
    pdf_path = report_gen.generate_pdf_report(
        result=result,
        column=column,
        model=model,
        forecast_days=forecast_days,
        username=session.get("user")
    )
    
    if os.path.exists(pdf_path):
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype="application/pdf"
        )
    else:
        flash("Error generating PDF report", "error")
        return redirect("/advanced/report")


@advanced_analysis_bp.route("/history")
def get_analysis_history():
    """Get user's analysis history"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session.get("user")
    history_file = os.path.join(BASE_DIR, "data", f"analysis_history_{username}.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        return jsonify({"history": history[-10:]})
    else:
        return jsonify({"history": []})


# ============================================================
# RAG CHATBOT ENDPOINT
# ============================================================

@advanced_analysis_bp.route("/rag-query", methods=["POST"])
def rag_analysis_query():
    """Ask questions about the analysis report"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"reply": "Please ask a question about your analysis."})
        
        # Get analysis context
        context = session.get("advanced_analysis_context", {})
        
        if not context:
            return jsonify({"reply": "No analysis found. Please run a prediction first, then I can answer questions about it."})
        
        # Generate RAG response
        reply = generate_rag_analysis_response(question, context)
        
        return jsonify({"reply": reply})
        
    except Exception as e:
        print(f"RAG query error: {e}")
        return jsonify({"reply": f"Sorry, I encountered an error: {str(e)}"}), 500


# Helper function to redirect
from flask import redirect