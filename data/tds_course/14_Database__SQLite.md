Database: SQLite
Relational databases are used to store data in a structured way. You’ll often access databases created by others for analysis.
PostgreSQL, MySQL, MS SQL, Oracle, etc. are popular databases. But the most installed database is 
SQLite
. It’s embedded into many devices and apps (e.g. your phone, browser, etc.). It’s lightweight but very scalable and powerful.
Watch these introductory videos to understand SQLite and how it’s used in Python (34 min):
There are many non-relational databases (NoSQL) like 
ElasticSearch
, 
MongoDB
, 
Redis
, etc. that you should know about and we may cover later.
Core Concepts:
-- Create a table


CREATE
 
TABLE
 users 
(

    id 
INTEGER
 
PRIMARY
 
KEY
,

    name 
TEXT
 
NOT
 
NULL
,

    email 
TEXT
 
UNIQUE
,

    created_at 
DATETIME
 
DEFAULT
 
CURRENT_TIMESTAMP


)
;



-- Insert data


INSERT
 
INTO
 users 
(
name
,
 email
)
 
VALUES

    
(
'Alice'
,
 
'alice@example.com'
)
,

    
(
'Bob'
,
 
'bob@example.com'
)
;



-- Query data


SELECT
 name
,
 
COUNT
(
*
)
 
as
 count

FROM
 users

GROUP
 
BY
 name

HAVING
 count 
>
 
1
;



-- Join tables


SELECT
 u
.
name
,
 o
.
product

FROM
 users u

LEFT
 
JOIN
 orders o 
ON
 u
.
id 
=
 o
.
user_id

WHERE
 o
.
status
 
=
 
'pending'
;
Copy to clipboard
Error
Copied
Python Integration:
import
 sqlite3

from
 pathlib 
import
 Path

import
 pandas 
as
 pd


async
 
def
 
query_database
(
db_path
:
 Path
,
 query
:
 
str
)
 
-
>
 pd
.
DataFrame
:

    
"""Execute SQL query and return results as DataFrame.

    Args:
        db_path: Path to SQLite database
        query: SQL query to execute

    Returns:
        DataFrame with query results
    """

    
try
:

        conn 
=
 sqlite3
.
connect
(
db_path
)

        
return
 pd
.
read_sql_query
(
query
,
 conn
)

    
finally
:

        conn
.
close
(
)



# Example usage

db 
=
 Path
(
'data.db'
)

df 
=
 
await
 query_database
(
db
,
 
'''
    SELECT date, COUNT(*) as count
    FROM events
    GROUP BY date
'''
)
Copy to clipboard
Error
Copied
Common Operations:
Database Management
-- Backup database


.
backup
 
'backup.db'



-- Import CSV


.
mode
 csv

.
import
 
data
.
csv table_name


-- Export results


.
headers 
on


.
mode
 csv

.
output results
.
csv

SELECT
 
*
 
FROM
 
table
;
Copy to clipboard
Error
Copied
Performance Optimization
-- Create index


CREATE
 
INDEX
 idx_user_email 
ON
 users
(
email
)
;



-- Analyze query


EXPLAIN
 QUERY 
PLAN


SELECT
 
*
 
FROM
 users 
WHERE
 email 
LIKE
 
'%@example.com'
;



-- Show indexes


SELECT
 
*
 
FROM
 sqlite_master 
WHERE
 
type
=
'index'
;
Copy to clipboard
Error
Copied
Data Analysis
-- Time series aggregation


SELECT

    
date
(
timestamp
)
,

    
COUNT
(
*
)
 
as
 events
,

    
AVG
(
duration
)
 
as
 avg_duration

FROM
 events

GROUP
 
BY
 
date
(
timestamp
)
;



-- Window functions


SELECT
 
*
,

    
AVG
(
amount
)
 
OVER
 
(

        
PARTITION
 
BY
 user_id
        
ORDER
 
BY
 
date

        
ROWS
 
BETWEEN
 
3
 
PRECEDING
 
AND
 
CURRENT
 
ROW

    
)
 
as
 moving_avg

FROM
 
transactions
;
Copy to clipboard
Error
Copied
Tools to work with SQLite:
SQLiteStudio
: Lightweight GUI
DBeaver
: Full-featured GUI
sqlite-utils
: CLI tool
Datasette
: Web interface














Previous




Spreadsheet: Excel, Google Sheets












Next










Version Control: Git, GitHub





