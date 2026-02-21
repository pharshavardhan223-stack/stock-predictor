import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_report(to_email, subject, body):

    sender_email = "mymailforn8n@gmail.com"      # YOUR EMAIL
    sender_password = "pharshavardhan@184"   # APP PASSWORD

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        server.login(sender_email, sender_password)

        server.send_message(msg)

        server.quit()

        print("✅ Email sent successfully!")

    except Exception as e:
        print("❌ Email Error:", e)
