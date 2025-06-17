Google Authentication with FastAPI
Secure your API endpoints using Google ID tokens to restrict access to specific email addresses.
Google Auth is the most commonly implemented single sign-on mechanism because:
It’s popular and user-friendly. Users can log in with their existing Google accounts.
It’s secure: Google supports OAuth2 and OpenID Connect to handle authentication.
Here’s how you build a FastAPI app that identifies the user.
Go to the 
Google Cloud Console – Credentials
 and click 
Create Credentials > OAuth client ID
.
Choose 
Web application
, set your authorized redirect URIs (e.g., 
http://localhost:8000/
).
Copy the 
Client ID
 and 
Client Secret
 into a 
.env
 file:
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
Copy to clipboard
Error
Copied
Create your FastAPI 
app.py
:
# /// script


# dependencies = ["python-dotenv", "fastapi", "uvicorn", "itsdangerous", "httpx", "authlib"]


# ///



import
 os

from
 dotenv 
import
 load_dotenv

from
 fastapi 
import
 FastAPI
,
 Request

from
 fastapi
.
responses 
import
 RedirectResponse

from
 starlette
.
middleware
.
sessions 
import
 SessionMiddleware

from
 authlib
.
integrations
.
starlette_client 
import
 OAuth

load_dotenv
(
)

app 
=
 FastAPI
(
)

app
.
add_middleware
(
SessionMiddleware
,
 secret_key
=
"create-a-random-secret-key"
)


oauth 
=
 OAuth
(
)

oauth
.
register
(

    name
=
"google"
,

    client_id
=
os
.
getenv
(
"GOOGLE_CLIENT_ID"
)
,

    client_secret
=
os
.
getenv
(
"GOOGLE_CLIENT_SECRET"
)
,

    server_metadata_url
=
"https://accounts.google.com/.well-known/openid-configuration"
,

    client_kwargs
=
{
"scope"
:
 
"openid email profile"
}
,


)



@app
.
get
(
"/"
)


async
 
def
 
application
(
request
:
 Request
)
:

    user 
=
 request
.
session
.
get
(
"user"
)

    
# 3. For authenticated users: say hello

    
if
 user
:

        
return
 
f"Hello 
{
user
[
'email'
]
}
"

    
# 2. For users who have just logged in, save their details in the session

    
if
 
"code"
 
in
 request
.
query_params
:

        token 
=
 
await
 oauth
.
google
.
authorize_access_token
(
request
)

        request
.
session
[
"user"
]
 
=
 token
[
"userinfo"
]

        
return
 RedirectResponse
(
"/"
)

    
# 1. For users who are logging in for the first time, redirect to Google login

    
return
 
await
 oauth
.
google
.
authorize_redirect
(
request
,
 request
.
url
)



if
 __name__ 
==
 
"__main__"
:

    
import
 uvicorn
    uvicorn
.
run
(
app
,
 port
=
8000
)
Copy to clipboard
Error
Copied
Now, run 
uv run app.py
.
When you visit 
http://localhost:8000/
 you’ll be redirected to a Google login page.
When you log in, you’ll be redirected back to 
http://localhost:8000/
Now you’ll see the email ID you logged in with.
Instead of displaying the email, you can show different content based on the user. For example:
Allow access to specfic users and not others
Fetch the user’s personalized information
Display different content based on the user














Previous




Web Framework: FastAPI












Next










Local LLMs: Ollama





