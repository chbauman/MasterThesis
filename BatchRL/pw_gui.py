from typing import Tuple

import wx


class LoginHolder:
    """
    Class that holds the login information.
    """

    def __init__(self):
        self.log_data = None


class LoginDialog(wx.Dialog):
    """
    Class to define login dialog
    Adapted from: https://dzone.com/articles/wxpython-how-create-login
    """

    def __init__(self, lh: LoginHolder):
        """Constructor"""

        wx.Dialog.__init__(self, None, title="Login")
        self.lh = lh

        # User info        
        user_sizer = wx.BoxSizer(wx.HORIZONTAL)
        user_lbl = wx.StaticText(self, label="Username:")
        user_sizer.Add(user_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.user = wx.TextCtrl(self)
        user_sizer.Add(self.user, 0, wx.ALL, 5)

        # Password info
        p_sizer = wx.BoxSizer(wx.HORIZONTAL)
        p_lbl = wx.StaticText(self, label="Password:")
        p_sizer.Add(p_lbl, 0, wx.ALL | wx.CENTER, 5)
        self.password = wx.TextCtrl(self, style=wx.TE_PASSWORD | wx.TE_PROCESS_ENTER)
        p_sizer.Add(self.password, 0, wx.ALL, 5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(user_sizer, 0, wx.ALL, 5)
        main_sizer.Add(p_sizer, 0, wx.ALL, 5)

        btn = wx.Button(self, label="Login")
        btn.Bind(wx.EVT_BUTTON, self.on_login)
        main_sizer.Add(btn, 0, wx.ALL | wx.CENTER, 5)

        self.SetSizer(main_sizer)

    def on_login(self, event) -> None:
        """
        Save login data to login holder.
        """

        user_password = self.password.GetValue()
        username = self.user.GetValue()
        self.lh.log_data = (username, user_password)
        self.Destroy()


class MainFrame(wx.Frame):
    """
    Frame calling the login dialog only.
    """

    def __init__(self, lh: LoginHolder):
        """Constructor"""
        self.lh = lh
        wx.Frame.__init__(self, None, title="Main App")
        wx.Panel(self)

        # Ask user to login
        dlg = LoginDialog(self.lh)
        dlg.ShowModal()
        self.Destroy()


def get_pw() -> Tuple[str, str]:
    """
    Opens GUI and retrieves the login information
    (username, password) and returns them in a tuple.
    """

    lh = LoginHolder()
    app = wx.App(False)
    frame = MainFrame(lh)

    # Run GUI
    frame.Show(False)
    app.MainLoop()
    if frame:
        frame.Destroy()

    return lh.log_data
