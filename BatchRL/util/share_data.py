import os
import shutil
import zipfile

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFile

from util.util import TEMP_DIR

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def g_drive_login() -> GoogleDrive:
    """Login to Google Drive and create and return drive object."""
    g_login = GoogleAuth()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)
    print("Authentication successful")
    return drive


def upload_folder_zipped(f_path, out_file_name: str = None):
    """Uploads the content of the folder `f_path` to Google Drive.

    If `out_file_name` is specified, this will be the name of the
    uploaded file, otherwise the name of the folder will be used.
    """
    f_name = os.path.basename(f_path)
    if out_file_name is None:
        out_file_name = f_name
    out_path = os.path.join(TEMP_DIR, out_file_name)
    shutil.make_archive(out_path, 'zip', f_path)
    upload_file(out_path + ".zip")


def download_and_extract_zipped_folder(base_name: str, extract_dir: str):
    f_name = base_name + ".zip"

    # Find file on Drive
    f_list, drive = get_root_files()
    found = [f for f in f_list if f["title"] == f_name]
    if len(found) > 1:
        print("Found multiple files, choosing first.")
    elif len(found) == 0:
        raise FileNotFoundError(f"No such file found: {f_name}")
    f = found[0]

    # Download to Temp folder
    out_temp_path = os.path.join(TEMP_DIR, f_name)
    f.GetContentFile(out_temp_path)

    # Unzip into folder
    with zipfile.ZipFile(out_temp_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def get_root_files():
    drive = g_drive_login()

    # Auto-iterate through all files in the root folder.
    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()

    return file_list, drive


def _rec_list(parent_dir: GoogleDriveFile, drive: GoogleDrive, lvl: int = 0):
    par_id = parent_dir["id"]
    ind = " " * 4 * lvl
    if parent_dir["mimeType"] == FOLDER_MIME_TYPE:
        # Found folder, recursively iterate over children.
        print(f"{ind}Folder: {parent_dir['title']}")
        file_list = drive.ListFile({'q': f"'{par_id}' in parents and trashed=false"}).GetList()
        for f in file_list:
            _rec_list(f, drive, lvl + 1)
    else:
        # Found file
        print(f"{ind}File: {parent_dir['title']}")


def list_files_recursively() -> None:
    """Lists the whole content of your Google Drive recursively.

    This is extremely slow!"""
    file_list, drive = get_root_files()

    # Iterate over all found files.
    for file1 in file_list:
        _rec_list(parent_dir=file1, drive=drive)


def upload_file(file_path, folder: str = None):
    """Uploads a file to Google Drive.

    If `folder` is not None, a folder with that name
    will be created and the file will be put into it.
    """
    drive = g_drive_login()

    if folder is not None:
        assert type(folder) == str
        # Create folder.
        folder_metadata = {
            'title': folder,
            # The mimetype defines this new file as a folder, so don't change this.
            'mimeType': FOLDER_MIME_TYPE,
        }
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        print("Uploaded Folder.")

    # Create file on drive.
    fn = os.path.basename(file_path)
    if folder is None:
        f = drive.CreateFile({'title': fn})
    else:
        assert isinstance(folder, GoogleDriveFile)
        folder_id = folder["id"]
        f = drive.CreateFile({"title": fn, "parents": [{"kind": "drive#fileLink", "id": folder_id}]})

    # Set and upload content.
    f.SetContentFile(file_path)
    f.Upload()
    print(f"The file: {file_path} has been uploaded")


def test_file_upload():
    """This is slow and requires user interaction."""
    TEST_DATA_DIR = "./tests/data"
    local_test_file = os.path.join(TEST_DATA_DIR, "test_upload_file.txt")
    upload_file(local_test_file, folder="test")
