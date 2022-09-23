from twilio.rest import Client

def send_message(content):
    account_sid = 'AC053b76a9be067ccef3e855b555478319'    
    auth_token = '45db360c0db0aa20d5e7f6ca17052322'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
            messaging_service_sid='MG3a57a37a69d41baa2799029154eb5aed', 
            body=content,
            to = '+8619924685689'
            )