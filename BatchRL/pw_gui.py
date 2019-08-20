# check for password entry, close application if not correct
import wx

# https://www.daniweb.com/programming/software-development/threads/370496/wxpython-gui-password-program

class MyClass(wx.Frame):
    def __init__(self, log_holder, parent=None, id=-1):
        wx.Frame.__init__(self, parent, id, 'Password Program', size=(300,300))
        panel = wx.Panel(self)
        self.lh = log_holder
        box = wx.TextEntryDialog(None, 'Please type your password:','Password')
        if box.ShowModal() == wx.ID_OK:
            answer = box.GetValue()
            box.Destroy()

        # check password, you want ot use 'rot13' to mask password
        if answer != 'password':

            # message to user            
            self.SetTitle('%s is incorrect password!' % answer)
            # wait 3 seconds, then close the app
            # or call dialog box again
            wx.FutureCall(3000, self.Destroy)

        # here goes the code to go on ...
        self.lh.log_data = [answer]
        self.Destroy()
        
       
class login_holder:
    def __init__(self):
        self.log_data = None

def getPW():
    lh = login_holder()
    app = wx.App(0)
    frame = MyClass(lh)
    frame.Show(False)
    app.MainLoop()
    if frame:
        frame.Destroy()

    log_data = lh.log_data
    print(log_data)
    return log_data