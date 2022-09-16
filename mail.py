import smtplib
from email.mime.text import MIMEText
from email.header import Header

import requests
import json
import time

mail_host = "smtp.qq.com"  
mail_user = "1030771474@qq.com" 
mail_pass = "wuqlccpikpcsbcfj"
sender = '1030771474@qq.com'
receivers = ['hotfox2001@hotmail.com'] 


def send_email(content):
    mail_content = "111"
    message = MIMEText(mail_content, 'plain', 'utf-8')
    message['Subject'] = Header('python发邮件', 'utf-8')
    message['From'] = Header("eez195", 'utf-8')
    message['To'] = Header("yq", 'utf-8')

    try:
        smtp = smtplib.SMTP_SSL(mail_host) # SMTP_SSL默认使用465端口
        #smtp.set_debuglevel(2)
        #smtp.ehlo(mail_host)
        smtp.login(mail_user, mail_pass)
        smtp.sendmail(sender, receivers, message.as_string()) # 发送邮件
        print("succeed")
    except smtplib.SMTPException:
        print("Error")