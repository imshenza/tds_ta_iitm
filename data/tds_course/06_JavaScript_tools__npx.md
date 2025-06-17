JavaScript tools: npx
npx
 is a command-line tool that comes with npm (Node Package Manager) and allows you to execute npm package binaries and run one-off commands without installing them globally. It’s essential for modern JavaScript development and data science workflows.
For data scientists, npx is useful when:
Running JavaScript-based data visualization tools
Converting notebooks and documents
Testing and formatting code
Running development servers
Here are common npx commands:
# Run a package without installing

npx http-server 
.
                
# Start a local web server

npx prettier 
--write
 
.
           
# Format code or docs

npx eslint 
.
                     
# Lint JavaScript

npx typescript-node script.ts    
# Run TypeScript directly

npx esbuild app.js               
# Bundle JavaScript

npx jsdoc 
.
                      
# Generate JavaScript docs



# Run specific versions

npx prettier@3.2 
--write
 
.
        
# Use prettier 3.2



# Execute remote scripts (use with caution!)

npx github:user/repo            
# Run from GitHub
Copy to clipboard
Error
Copied
Watch this introduction to npx (6 min):














Previous




Python tools: uv












Next










Unicode





