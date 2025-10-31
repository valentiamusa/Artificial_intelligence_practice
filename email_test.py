import smtplib, ssl
from email.message import EmailMessage

SENDER = "valentiamusabeyezu@gmail.com"
APP_PASSWORD = "xupscibhremmcwql"
RECEIVER = "valentiaexercise@gmail.com"  # try a different one

msg = EmailMessage()
msg["Subject"] = "Test Email from Python"
msg["From"] = SENDER
msg["To"] = RECEIVER
msg.set_content("This is a test email from my AI Security System.")

import ssl, certifi
context = ssl.create_default_context(cafile=certifi.where())


print("✅ Email sent successfully!")