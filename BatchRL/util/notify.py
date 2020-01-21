import smtplib
import ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "chris.python.notifyer@gmail.com"  # Enter your address
receiver_email = "chris.baum.1995@gmail.com"  # Enter receiver address

message = """\
Subject: Hi there

This message is sent from Python."""


def send_mail():
    password = input("Type your password and press enter: ")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
