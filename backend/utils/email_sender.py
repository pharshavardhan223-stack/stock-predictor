import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# CONFIG (CHANGE THESE)
EMAIL_ADDRESS = "mymailforn8n@gmail.com"
EMAIL_PASSWORD = "pharshavardhan@184"


def send_report(to_email, subject, body):

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

        server.send_message(msg)
        server.quit()

        return True

    except Exception as e:
        print("Email Error:", e)
        return False
