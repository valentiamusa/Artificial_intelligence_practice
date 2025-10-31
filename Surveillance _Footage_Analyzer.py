 
import ctypes
from email.message import EmailMessage
import winsound
from ultralytics import YOLO
import cv2
from datetime import datetime
import os
import smtplib, ssl
import certifi
 


# --- Email Config ---
SENDER = "valentiamusabeyezu@gmail.com"
APP_PASSWORD = "xupscibhremmcwql"
RECEIVER = "valentiaexercise@gmail.com"

def send_email_alert(image_path):
    if not os.path.exists(image_path):
        print("[WARN] Image not found, skipping email.")
        return

    msg = EmailMessage()
    msg["Subject"] = "🚨 Security Alert: Person Detected"
    msg["From"] = SENDER
    msg["To"] = RECEIVER
    msg.set_content("A person was detected by your AI Security System. See the attached image.")

    with open(image_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    # ✅  Use trusted Mozilla CA bundle from certifi
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())


    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
    # login and send logic

        smtp.login(SENDER, APP_PASSWORD)
        smtp.send_message(msg)

    print(f"[EMAIL SENT] Alert sent with image: {image_path}")
    
# ---------------- SETUP ----------------
os.makedirs("captures", exist_ok=True)
model = YOLO("yolov8n.pt")     # pretrained YOLOv8 model
cap = cv2.VideoCapture(0)

# read two frames for motion comparison
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # --- MOTION DETECTION ---
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = any(cv2.contourArea(c) > 2000 for c in contours)

    # --- WHEN MOTION IS DETECTED, RUN YOLO ---
    if motion_detected:
        results = model(frame1, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                # draw boxes + labels
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame1, f"{label} {conf:.2f}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # save evidence
                if conf > 0.7:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"captures/{label}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame1)
                    print(f"[{timestamp}] Motion: {label} ({conf:.2f}) saved!")
                if label.lower() == "person":
    # --- Save image of detection ---
                     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                     image_path = f"captures/person_{timestamp}.jpg"
                     cv2.imwrite(image_path, frame1)
                     print(f"[ALERT] {timestamp}: PERSON DETECTED ({conf:.2f})")

    # --- Beep and show popup ---
                     winsound.Beep(1000, 400)
                     ctypes.windll.user32.MessageBoxW(0, "🚨 PERSON DETECTED!", "SECURITY ALERT", 1)

    # --- Send email alert with image attachment ---
                     send_email_alert(image_path)
                     print("[INFO] Email alert sent successfully (attempted).")
                           # Send Email
                        
    # show camera feed
    cv2.imshow("YOLOv8 Smart Surveillance", frame1)

    # update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
