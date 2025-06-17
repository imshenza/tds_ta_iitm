Web Framework: FastAPI
FastAPI
 is a modern Python web framework for building APIs with automatic interactive documentation. It’s fast, easy to use, and designed for building production-ready REST APIs.
Here’s a minimal FastAPI app, 
app.py
:
# /// script


# requires-python = ">=3.11"


# dependencies = [


#   "fastapi",


#   "uvicorn",


# ]


# ///



from
 fastapi 
import
 FastAPI

app 
=
 FastAPI
(
)



@app
.
get
(
"/"
)


async
 
def
 
root
(
)
:

    
return
 
{
"message"
:
 
"Hello!"
}



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
 host
=
"0.0.0.0"
,
 port
=
8000
)
Copy to clipboard
Error
Copied
Run this with 
uv run app.py
.
Handle errors by raising HTTPException
from
 fastapi 
import
 HTTPException


async
 
def
 
get_item
(
item_id
:
 
int
)
:

    
if
 
not
 valid_item
(
item_id
)
:

        
raise
 HTTPException
(

            status_code
=
404
,

            detail
=
f"Item 
{
item_id
}
 not found"

        
)
Copy to clipboard
Error
Copied
Use middleware for logging
from
 fastapi 
import
 Request

import
 time


@app
.
middleware
(
"http"
)


async
 
def
 
add_timing
(
request
:
 Request
,
 call_next
)
:

    start 
=
 time
.
time
(
)

    response 
=
 
await
 call_next
(
request
)

    response
.
headers
[
"X-Process-Time"
]
 
=
 
str
(
time
.
time
(
)
 
-
 start
)

    
return
 response
Copy to clipboard
Error
Copied
Tools:
FastAPI CLI
: Project scaffolding
Pydantic
: Data validation
SQLModel
: SQL databases
FastAPI Users
: Authentication
Watch this FastAPI Course for Beginners (64 min):














Previous




REST APIs












Next










Authentication: Google Auth





