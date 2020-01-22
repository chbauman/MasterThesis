"""Notification module.

Can be used to send mails from your gmail account.
You need to allow unsave apps access under your
account settings.
"""

import os
import smtplib
import ssl
from pathlib import Path
from typing import List

PORT = 465  #: Port for SSL
smtp_server: str = "smtp.gmail.com"
sender_email: str = "chris.python.notifyer@gmail.com"  #: Sender address
debug_email: str = "chris.python.debug@gmail.com"  #: Debug receiver address
receiver_email: str = "chris.baum.1995@gmail.com"  #: Real receiver address


curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
pw_def_path = os.path.join(curr_dir.parent.parent, "python_notifyer.txt")


def login_from_file(file_name: str) -> List[str]:
    """Loads login information from a file."""
    assert os.path.isfile(file_name), f"File: {file_name} not found!"
    with open(file_name, "r") as f:
        return [l.rstrip() for l in f if l.rstrip() != ""]


def _pw_from_file(file_name: str = pw_def_path) -> str:
    """Reads the password in the file given."""
    login = login_from_file(file_name)
    assert len(login) == 1, f"Invalid password: {login}"
    return login[0]


def send_mail(debug: bool = True,
              subject: str = "Hello there!",
              msg: str = "General Kenobi") -> None:
    """Sends a mail via python.

    Args:
        debug: Whether to use debug mode, will send the mail to the
            debug address.
        subject: Subject of the mail.
        msg: Message of the mail.

    Returns:

    """
    rec_mail = debug_email if debug else receiver_email

    message = f"""Subject: {subject}\n\n\
    {msg}\n\n
    This is an automatically generated message, do not reply!"""

    password = _pw_from_file()
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, PORT, context=context) as server:
        server.login(sender_email, password)
        send_errs = server.sendmail(sender_email, rec_mail, message)
        if len(send_errs) > 0:
            print(f"Errors happened: {send_errs}")
