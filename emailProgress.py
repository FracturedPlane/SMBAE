
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
##################################
# You can start the email server like so
#  python -m smtpd -n -c DebuggingServer localhost:1025
# Better yet, you should install sendmail

import sys, json


def send_mail(send_from, send_to, subject, text, files=None,
              server="127.0.0.1"):
    assert isinstance(send_to, list)

    msg = MIMEMultipart(
        From=send_from,
        To=COMMASPACE.join(send_to),
        Date=formatdate(localtime=True),
        Subject=subject
    )
    text = 'Subject: %s\n\n%s' % (subject, text)
    msg['Subject'] = subject
    msg.attach(MIMEText(text))
    msg.preamble = subject

    for f in files or []:
        with open(f, "rb") as fil:
            msg.attach(MIMEApplication(
                fil.read(),
                Content_Disposition='attachment; filename="%s"' % basename(f),
                Name=basename(f)
            ))

    smtp = smtplib.SMTP(server)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
    
    
def sendData(settings):
    from util.SimulationUtil import getDataDirectory, getAgentName
    
    
    from_email="admin@fracturedplane.com"
    email_subject="Simulation Data"
    email_text="""
    This email includes some data on the current state of the simulation. \n
    
    Take care.\n
    """
    directory = getDataDirectory(settings)
    agentName = getAgentName(settings)
    print ("Data folder: ", directory)
    
    trainingGraph=directory+agentName+'_'+".png"
    try:
        send_mail(send_from=from_email,send_to=['glen@fracturedplane.com'], 
            subject=email_subject, text=email_text, files=[trainingGraph])
    except Exception as e:
        print 'Error email data: %s' % e    
    except:
        print "Emailling of simulation data failed "
        print "Unexpected error:", sys.exc_info()[0]
        pass
    
    
    
    
    
if __name__ == '__main__':
    
    print ("Something")
    settingsFileName = sys.argv[1]
    file = open(settingsFileName)
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings)))
    file.close()
    import os
    sendData(settings)