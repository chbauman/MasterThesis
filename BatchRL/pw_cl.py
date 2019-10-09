from typing import Tuple


def get_pw() -> Tuple[str, str]:
    """
    Commandline login getter tool.
    """
    username = input("Please enter username: ")
    pw = input("Enter password: ")
    return username, pw
