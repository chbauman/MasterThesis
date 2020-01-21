import os
import smtplib
import ssl
from pathlib import Path

port = 465  #: For SSL
smtp_server: str = "smtp.gmail.com"
sender_email: str = "chris.python.notifyer@gmail.com"  #: Sender address
debug_email: str = "chris.python.debug@gmail.com"  #: Debug receiver address
receiver_email: str = "chris.baum.1995@gmail.com"  #: Real receiver address


curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
pw_def_path = os.path.join(curr_dir.parent.parent, "python_notifyer.txt")


def _pw_from_file(file_name: str = pw_def_path):
    with open(file_name, "r") as f:
        return f.read()


def send_mail(debug: bool = True,
              subject: str = "Hello there!",
              msg: str = "General Kenobi"):
    rec_mail = debug_email if debug else receiver_email

    message = f"""Subject: {subject}\n\n\
    {msg}"""

    password = _pw_from_file()
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, rec_mail, message)
