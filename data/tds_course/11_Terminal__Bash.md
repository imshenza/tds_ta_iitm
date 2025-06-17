Terminal: Bash
UNIX shells are the de facto standard in the data science world and 
Bash
 is the most popular.
This is available by default on Mac and Linux.
On Windows, install 
Git Bash
 or 
WSL
 to get a UNIX shell.
Watch this video to install WSL (12 min).
Watch this video to understand the basics of Bash and UNIX shell commands (75 min).
Essential Commands:
# File Operations


ls
 
-la
               
# List all files with details


cd
 path/to/dir       
# Change directory


pwd
                  
# Print working directory


cp
 
source
 dest       
# Copy files


mv
 
source
 dest       
# Move/rename files


rm
 
-rf
 
dir
           
# Remove directory recursively



# Text Processing


grep
 
"pattern"
 
file
  
# Search for pattern


sed
 
's/old/new/'
 f   
# Replace text


awk
 
'{print $1}'
 f   
# Process text by columns


cat
 
file
 
|
 
wc
 
-l
     
# Count lines



# Process Management


ps
 aux               
# List processes


kill
 
-9
 PID          
# Force kill process


top
                  
# Monitor processes


htop
                 
# Interactive process viewer



# Network


curl
 url             
# HTTP requests


wget
 url             
# Download files


nc
 
-zv
 
host
 port     
# Test connectivity


ssh
 user@host        
# Remote login



# Count unique values in CSV column


cut
 -d
','
 
-f1
 data.csv 
|
 
sort
 
|
 
uniq
 
-c



# Quick data analysis


awk
 -F
','
 
'{sum+=$2} END {print sum/NR}'
 data.csv  
# Average


sort
 -t
','
 
-k2
 
-n
 data.csv 
|
 
head
                  
# Top 10



# Monitor log in real-time


tail
 
-f
 log.txt 
|
 
grep
 
--color
 
'ERROR'
Copy to clipboard
Error
Copied
Bash Scripting Essentials:
#!/bin/bash



# Variables


NAME
=
"value"


echo
 
$NAME



# Loops


for
 
i
 
in
 
{
1
..
5
}
;
 
do

    
echo
 
$i


done



# Conditionals


if
 
[
 
-f
 
"file.txt"
 
]
;
 
then

    
echo
 
"File exists"


fi



# Functions


process_data
(
)
 
{

    
local
 
input
=
$1

    
echo
 
"Processing 
$input
"


}
Copy to clipboard
Error
Copied
Productivity Tips:
Command History
history
         
# Show command history

Ctrl+R         
# Search history


!
!
             
# Repeat last command


!
$             
# Last argument
Copy to clipboard
Error
Copied
Directory Navigation
pushd
 
dir
      
# Push directory to stack


popd
           
# Pop directory from stack


cd
 -           
# Go to previous directory
Copy to clipboard
Error
Copied
Job Control
command
 
&
      
# Run in background

Ctrl+Z         
# Suspend process


bg
             
# Resume in background


fg
             
# Resume in foreground
Copy to clipboard
Error
Copied
Useful Aliases
 - typically added to 
~/.bashrc
alias
 
ll
=
'ls -la'


alias
 
gs
=
'git status'


alias
 
jupyter
=
'jupyter notebook'


alias
 
activate
=
'source venv/bin/activate'
Copy to clipboard
Error
Copied














Previous




JSON












Next










AI Terminal Tools: llm





