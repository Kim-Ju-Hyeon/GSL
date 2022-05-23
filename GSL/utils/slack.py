import requests
import datetime

def get_slack_token():
        token_file = open('../../slack_token', 'r')
        token = token_file.read()
        return token

def send_slack_message(message):
        token = get_slack_token()
        myToken = token
        channel_id = "C03GANY5YSJ"

        data = {'content-Type': 'application/x-www-form-urlencoded',
                'token': myToken,
                'channel': channel_id,
                'text': message
                }

        URL = "https://slack.com/api/chat.postMessage"
        requests.post(URL, data=data)


def convert_seconds_to_kor_time(in_seconds):
        """초를 입력받아 읽기쉬운 한국 시간으로 변환"""
        t1 = in_seconds
        days = t1.days
        _sec = t1.seconds
        (hours, minutes, seconds) = str(datetime.timedelta(seconds=_sec)).split(':')
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)

        result = []
        if days >= 1:
                result.append(str(days) + '일')
        if hours >= 1:
                result.append(str(hours) + '시간')
        if minutes >= 1:
                result.append(str(minutes) + '분')
        if seconds >= 1:
                result.append(str(seconds) + '초')
        return ' '.join(result)

def slack_message(start, message):
        end = datetime.datetime.now()
        end = end + datetime.timedelta(hours=9)
        string_k = convert_seconds_to_kor_time(end - start)
        
        send_slack_message(message + f"\n It takes {string_k}")