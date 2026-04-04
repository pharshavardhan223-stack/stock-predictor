"""
FILE: backend/utils/report_generator.py
PURPOSE: Generate professional PDF reports from analysis results
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

# Create directories if they don't exist
os.makedirs(REPORT_DIR, exist_ok=True)


class ReportGenerator:
    """Generate professional PDF reports for analysis"""
    
    def __init__(self):
        self.report_dir = REPORT_DIR
        print(f"Report directory: {self.report_dir}")
    
    def generate_pdf_report(self, result, column, model, forecast_days, username):
        """Generate complete PDF report"""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{username}_{timestamp}.pdf"
        filepath = os.path.join(self.report_dir, filename)
        
        print(f"Generating report: {filepath}")
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=letter, 
                                topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                     fontSize=24, textColor=colors.HexColor('#003366'), 
                                     alignment=TA_CENTER, spaceAfter=20)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                       fontSize=14, textColor=colors.HexColor('#0288d1'), 
                                       spaceAfter=10, spaceBefore=10)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], 
                                      fontSize=10, spaceAfter=6)
        
        # Title
        story.append(Paragraph(f"Advanced Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("📊 EXECUTIVE SUMMARY", heading_style))
        
        current_val = result.get("current_value", 0)
        predicted_val = result.get("predicted_value", 0)
        trend = result.get("trend", "stable")
        confidence = result.get("confidence", 0)
        
        model_names = {
            "linear_regression": "Linear Regression",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "ensemble": "Ensemble",
            "auto": "Auto (Best Model)"
        }
        model_name = model_names.get(model, "AI Model")
        
        summary_data = [
            ["Analyzed Column", column],
            ["Model Used", model_name],
            ["Current Value", f"{current_val:,.2f}"],
            ["Predicted Value", f"{predicted_val:,.2f}"],
            ["Expected Change", f"{result.get('trend_percent', 0):.1f}%"],
            ["Trend Direction", "📈 UP" if trend == "up" else "📉 DOWN" if trend == "down" else "➡️ STABLE"],
            ["Confidence Level", f"{confidence:.0f}%"],
            ["Recommendation", result.get("recommendation", "HOLD")]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#003366')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ccd9e8'))
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # Add chart (with error handling)
        chart_image = self._create_forecast_chart(result, column)
        if chart_image:
            try:
                # Use BytesIO to avoid file path issues
                img = Image(chart_image, width=6*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"Error adding chart to PDF: {e}")
                story.append(Paragraph("⚠️ Chart could not be generated", normal_style))
        
        # Statistics
        story.append(Paragraph("📈 STATISTICAL SUMMARY", heading_style))
        stats = result.get("statistics", {})
        stats_data = [
            ["Mean (Average)", f"{stats.get('mean', 0):,.2f}"],
            ["Median", f"{stats.get('median', 0):,.2f}"],
            ["Minimum", f"{stats.get('min', 0):,.2f}"],
            ["Maximum", f"{stats.get('max', 0):,.2f}"],
            ["Standard Deviation", f"{stats.get('std', 0):,.2f}"],
            ["Data Points", stats.get('data_points', 0)]
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd'))
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 15))
        
        # Forecast Table
        story.append(Paragraph("🔮 FORECAST (Next 10 Periods)", heading_style))
        predictions = result.get("predictions", [])[:10]
        forecast_dates = result.get("forecast_dates", [])[:10]
        
        if predictions and forecast_dates:
            forecast_data = [["Date", "Predicted Value", "Change"]]
            for i, (date, pred) in enumerate(zip(forecast_dates, predictions)):
                change = pred - (predictions[i-1] if i > 0 else result.get("current_value", 0))
                change_symbol = "▲" if change > 0 else "▼" if change < 0 else "→"
                forecast_data.append([date, f"{pred:,.2f}", f"{change_symbol} {abs(change):,.2f}"])
            
            forecast_table = Table(forecast_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0288d1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
            ]))
            story.append(forecast_table)
            story.append(Spacer(1, 15))
        
        # Insights
        if result.get("insights"):
            story.append(Paragraph("💡 AI-GENERATED INSIGHTS", heading_style))
            for insight in result["insights"]:
                story.append(Paragraph(insight, normal_style))
            story.append(Spacer(1, 10))
        
        # Model Performance
        story.append(Paragraph("📊 MODEL PERFORMANCE", heading_style))
        metrics = result.get("metrics", {})
        metrics_data = [
            ["Metric", "Value", "Interpretation"],
            ["Mean Absolute Error (MAE)", f"{metrics.get('mae', 0):,.2f}", "Lower is better"],
            ["Root Mean Square Error (RMSE)", f"{metrics.get('rmse', 0):,.2f}", "Lower is better"],
            ["R² Score", f"{metrics.get('r2', 0):.4f}", "1.0 = perfect fit"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a6080')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 15))
        
        # Anomalies
        anomalies = result.get("anomalies", [])
        if anomalies:
            story.append(Paragraph("⚠️ DETECTED ANOMALIES", heading_style))
            for a in anomalies[:5]:
                story.append(Paragraph(f"• Position {a['index']}: Value {a['value']:.2f} (Z-Score: {a['z_score']}) - {a['severity']} severity", normal_style))
            story.append(Spacer(1, 10))
        
        # Disclaimer
        story.append(Spacer(1, 20))
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], 
                                          fontSize=8, textColor=colors.HexColor('#999999'), 
                                          alignment=TA_CENTER)
        story.append(Paragraph("⚠️ DISCLAIMER: This report is AI-generated for informational purposes only. Not financial advice.", disclaimer_style))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF generated successfully: {filepath}")
        except Exception as e:
            print(f"Error building PDF: {e}")
            # Create a simple fallback PDF
            doc.build([Paragraph("Report could not be generated completely", normal_style)])
        
        return filepath
    
    def _create_forecast_chart(self, result, column):
        """Create forecast chart and return as BytesIO object"""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            predictions = result.get("predictions", [])
            forecast_dates = result.get("forecast_dates", [])
            current_value = result.get("current_value", 0)
            
            if not predictions or not forecast_dates:
                print("No predictions or dates for chart")
                return None
            
            # Convert dates
            dates = []
            for d in forecast_dates:
                try:
                    dates.append(datetime.strptime(d, "%Y-%m-%d"))
                except:
                    dates.append(datetime.now() + timedelta(days=len(dates)))
            
            # Plot
            ax.plot(dates, predictions, 'b-', linewidth=2, label='Forecast', marker='o', markersize=4)
            ax.fill_between(dates, predictions, alpha=0.2, color='blue')
            ax.axhline(y=current_value, color='r', linestyle='--', linewidth=1, label=f'Current: {current_value:,.2f}')
            
            ax.set_title(f'{column} - Forecast Chart', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to BytesIO instead of file
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            img_bytes.seek(0)
            
            print("✅ Chart created successfully as BytesIO")
            return img_bytes
            
        except Exception as e:
            print(f"Chart creation error: {e}")
            import traceback
            traceback.print_exc()
            return None