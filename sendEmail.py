# Import smtplib for the actual sending function
import smtplib
import socket
import sys
import getpass

# Import the email modules we'll need
from email.mime.text import MIMEText
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# with open(textfile) as fp:
def sendEmail(subject, contents, hyperSettings, simSettings=None, testing=False, dataFile=None, pictureFile=None):
    # Create a text/plain message
    messageBody = contents + "\n" + str(simSettings)
    msg = MIMEMultipart()
    msgBody = MIMEText(messageBody)
    msg.attach(msgBody)
    
    hostname = socket.getfqdn()
    
    print("Hostname: ", hostname)
    
    username = getpass.getuser()
    print ("username: ", username)
    
    fromEmail = hyperSettings['from_email_address']
    print("From email: ", fromEmail)
    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = subject
    # msg['From'] = 'csguestn@dyn.cs.sfu.ca'
    msg['From'] = fromEmail
    msg['To'] = 'gberseth@cs.ubc.ca'
    
    # print("Email message: ", msg)
    
    ### attach a compressed file
    if ( not (dataFile is None) ):
        fileName_ = dataFile
        fp = open(fileName_, 'rb')
        ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        msgFiles = MIMEBase(maintype, subtype)
        msgFiles.set_payload(fp.read())
        # Encode the payload using Base64
        encoders.encode_base64(msgFiles)
        # Set the filename parameter
        msgFiles.add_header('Content-Disposition', 'attachment', filename=fileName_)
        msg.attach(msgFiles)
    
    # Assume we know that the image files are all in PNG format
    if ( (pictureFile is not None) ):
        # for file in pngfiles:
        # Open the files in binary mode.  Let the MIMEImage class automatically
        # guess the specific image type.
        print ("Attaching image: ", pictureFile)
        fp = open(pictureFile, 'rb')
        img = MIMEImage(fp.read())
        fp.close()
        img.add_header('Content-Disposition', 'attachment', filename=pictureFile)
        msg.attach(img)
    
    if ( testing ):
        return
    # Send the message via our own SMTP server.
    s = smtplib.SMTP(hyperSettings['mail_server_name'])
    s.send_message(msg)
    s.quit()
    print ("Email sent.")


if __name__ == '__main__':
    import json
        
    print ("len(sys.argv): ", len(sys.argv))
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        sendEmail("Testing", "Nothing", settings=settings_, testing=True)
    elif (len(sys.argv) == 3):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        sendEmail("Testing", "Nothing", settings=settings_, testing=False)
    
    elif (len(sys.argv) == 4):
        settingsFileName = sys.argv[1] 
        file = open(settingsFileName)
        settings_ = json.load(file)
        print ("Settings: " + str(json.dumps(settings_)))
        file.close()
        
        sendEmail("Testing", "Nothing", settings=settings_, simSettings=sys.argv[2], testing=False)
        
    else:
        sendEmail("Testing", "Nothing", hyperSettings={}, testing=True)