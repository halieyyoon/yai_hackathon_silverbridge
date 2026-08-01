import os

from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
phone_number = os.getenv("TWILIO_PHONE_NUMBER")
public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

if not all([account_sid, auth_token, phone_number, public_base_url]):
    raise RuntimeError("Missing Twilio or PUBLIC_BASE_URL environment variables.")

client = Client(account_sid, auth_token)

numbers = client.incoming_phone_numbers.list(
    phone_number=phone_number,
    limit=1,
)

if not numbers:
    raise RuntimeError(
        "The number was not found as a programmable phone number in this account."
    )

number = numbers[0]

updated = client.incoming_phone_numbers(number.sid).update(
    voice_url=f"{public_base_url}/voice/incoming",
    voice_method="POST",
    status_callback=f"{public_base_url}/voice/status",
    status_callback_method="POST",
)

print("Webhook configured successfully")
print("Number:", updated.phone_number)
print("Voice URL:", updated.voice_url)