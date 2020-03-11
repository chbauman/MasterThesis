import os

NEST_LOGIN_FILE = "rest_login.txt"
OPCUA_LOGIN_FILE = "opcua_login.txt"
EMAIL_LOGIN_FILE = "notify_email_login.txt"
DEBUG_EMAIL_LOGIN_FILE = "notify_email_debug_login.txt"


def str2bool(v) -> bool:
    """Converts a string to a boolean.

    Raises:
        ValueError: If it cannot be converted.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', '1.0'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', '0.0'):
        return False
    else:
        raise ValueError(f"Boolean value expected, got {v}")


def get_login_and_write_to_file(file_name: str, name: str):

    # Check if login already exists
    f_exists = os.path.isfile(file_name)
    create_file = not f_exists
    if f_exists:
        parse_error = True
        while parse_error:
            ans = input(f"Overwrite {name} login info? ")
            try:
                create_file = str2bool(ans)
                parse_error = False
            except ValueError:
                print("Your input was not understood!")

    # Ask for login and save to file
    if create_file:
        nest_user = input(f"Provide your {name} username: ")
        nest_pw = input("And password: ")

        with open(file_name, "w") as f:
            f.write(nest_user + "\n")
            f.write(nest_pw + "\n")


def main():
    print("Setting up everything...")

    if not os.path.isdir("venv"):
        print("Setting up virtual environment...")
        # TODO

    # Get NEST login data and store in file
    get_login_and_write_to_file(NEST_LOGIN_FILE, "NEST database")

    # Get NEST login data and store in file
    get_login_and_write_to_file(OPCUA_LOGIN_FILE, "Opcua client")

    # Get notification email login data and store in file
    print("Setup done!")


if __name__ == '__main__':
    main()
