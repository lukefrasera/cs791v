#!/usr/bin/env python
import time
import os, sys
import smtplib as slib
from email.mime.text import MIMEText as mt

while True:
	try:
		os.kill(int(sys.argv[1]),0)
		time.sleep(10)

	except OSError:
		msg = mt("Process is Dead!: Data Collection Complete!")
		msg['Subject'] = 'Process has stopped'
		mail_from = 'fraser@nevada.unr.edu'
		msg['From'] = mail_from
		mail_to = 'lukefrasera@gmail.com'
		msg['To'] = mail_to
		s = slib.SMTP('smtp.gmail.com:587')
		s.ehlo()
		s.starttls()
		s.login('fraser@nevada.unr.edu', "123AdriaN?!';p aswzxc")
		s.sendmail(mail_from, mail_to, msg.as_string())
		s.quit()
		break
