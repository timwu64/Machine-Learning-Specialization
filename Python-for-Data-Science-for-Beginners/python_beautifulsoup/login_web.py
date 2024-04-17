import requests
import sys

EMAIL = ''
PASSWORD = ''

URL = 'http://friends.cisv.org'

def main():
    # Start a session so we can have persistant cookies
    session = requests.session(config={'verbose': sys.stderr})

    # This is the form data that the page sends when logging in
    login_data = {
        'loginemail': EMAIL,
        'loginpswd': PASSWORD,
        'submit': 'login',
    }

    # Authenticate
    r = session.post(URL, data=login_data)

    # Try accessing a page that requires you to be logged in
    r = session.get('http://friends.cisv.org/index.cfm?fuseaction=user.fullprofile')

if __name__ == '__main__':
    main()
