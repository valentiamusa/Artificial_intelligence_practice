import smtplib, ssl, certifi
from email.message import EmailMessage

SENDER = "valentiamusabeyezu@gmail.com"
APP_PASSWORD = "xupscibhremmcwql"
RECEIVER = "valentiaexercise@gmail.com"  # a different address, not the same Gmail

msg = EmailMessage()
msg["Subject"] = "Test from AI Security System"
msg["From"] = SENDER
msg["To"] = RECEIVER
msg.set_content("This is a Gmail test from Python with proper SSL certificates.")


context = ssl.create_default_context()
context.load_verify_locations(cafile=certifi.where())

with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
    # your login and email logic here





    smtp.login(SENDER, APP_PASSWORD)
    smtp.send_message(msg)

'''import ssl
print(ssl.get_default_verify_paths())

import sys
print(sys.executable)'''


print("✅ Email sent successfully!")
