"""Notification module.

Can be used to send mails from your gmail account.
You need to allow unsave apps access under your
account settings.
"""

import os
import smtplib
import ssl
import sys
import time
import traceback
from pathlib import Path
from typing import List

from util.util import force_decorator_factory

PORT = 465  #: Port for SSL
sender_email: str = "chris.python.notifyer@gmail.com"  #: Sender address
debug_email: str = "chris.python.debug@gmail.com"  #: Debug receiver address
receiver_email: str = "chris.baum.1995@gmail.com"  #: Real receiver address

curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
pw_def_path = os.path.join(curr_dir.parent.parent, "python_notifyer.txt")


def set_exit_handler(func):
    """Catching kill events.

    Should work for windows and linux.
    From: https://danielkaes.wordpress.com/2009/06/04/how-to-catch-kill-events-with-python/
    """
    if os.name == "nt":
        try:
            import win32api
            win32api.SetConsoleCtrlHandler(func, True)
        except ImportError:
            version = ".".join(map(str, sys.version_info[:2]))
            raise Exception("pywin32 not installed for Python " + version)
    else:
        import signal
        signal.signal(signal.SIGTERM, func)


def test_kill_event():
    """Test for catching kill events.

    You have 30 seconds to kill the execution and see what
    happens, then check your mail. :)
    """
    with FailureNotifier("test", verbose=0, debug=True):
        print("Sleeping...")
        time.sleep(30.0)
        print("Done Sleeping, you were too late!")
        raise ValueError("Fuck")


class FailureNotifier:
    """Context manager for failure notifications.

    Sends a mail if an error happens while it is active,
    sends the stack trace."""

    def __init__(self, name: str, verbose: int = 1,
                 debug: bool = False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def __enter__(self):
        if self.verbose:
            print("Entering FailureNotifier...")

        # Set the exit handler to the exit function since
        # e.g. if you press X on the powershell console, this
        # will not be caught by the context manager.
        def on_exit(sig, func=None):
            self._on_exit(msg=f"Program was mysteriously killed by somebody. "
                              f"Clues are signal: {sig} and func: {func}")

        set_exit_handler(on_exit)
        return self

    def _on_exit(self, msg):
        sub = f"Error while executing '{self.name}'."
        send_mail(self.debug, subject=sub,
                  msg=msg)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):

        p_msg = "Exiting FailureNotifier "
        if exc_type is not None:
            # Unhandled exception happened, notify owner.
            msg = traceback.format_exc()
            self._on_exit(msg=msg)
            p_msg += "with unhandled Error."
        else:
            p_msg += "successfully."

        if self.verbose:
            print(p_msg)


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


@force_decorator_factory()
def send_mail(debug: bool = True,
              subject: str = "Hello there!",
              msg: str = "General Kenobi",
              use_ssl: bool = True) -> None:
    """Sends a mail via python.

    Args:
        debug: Whether to use debug mode, will send the mail to the
            debug address.
        subject: Subject of the mail.
        msg: Message of the mail.
        use_ssl: Whether to use SSL, use default.
    """
    rec_mail = debug_email if debug else receiver_email

    message = f"Subject: {subject}\n\n{msg}\n\n" \
              f"This is an automatically generated message, do not reply!"

    port = 465 if use_ssl else 587  #: Port for SSL
    smtp_server = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    password = _pw_from_file()
    ssl.create_default_context()
    with smtp_server("smtp.gmail.com", port) as server:
        server.login(sender_email, password)
        send_errs = server.sendmail(sender_email, rec_mail, message)
        if len(send_errs) > 0:
            print(f"Error(s) happened: {send_errs}")
