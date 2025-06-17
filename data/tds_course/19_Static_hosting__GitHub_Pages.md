Static hosting: GitHub Pages
GitHub Pages
 is a free hosting service that turns your GitHub repository directly into a static website whenever you push it. This is useful for sharing analysis results, data science portfolios, project documentation, and more.
Common Operations:
# Create a new GitHub repo


mkdir
 my-site

cd
 my-site

git
 init


# Add your static content


echo
 
"<h1>My Site</h1>"
 
>
 index.html


# Push to GitHub


git
 
add
 
.


git
 commit 
-m
 
"feat(pages): initial commit"


git
 push origin main


# Enable GitHub Pages from the main branch on the repo settings page
Copy to clipboard
Error
Copied
Best Practices:
Keep it small
Optimize images
. Prefer SVG over WEBP over 8-bit PNG.
Preload
 critical assets like stylesheets
Avoid committing large files like datasets, videos, etc. directly. Explore 
Git LFS
 instead.
Tools:
GitHub Desktop
: GUI for Git operations
GitHub CLI
: Command line interface
GitHub Actions
: Automation














Previous




Images: Compression












Next










Notebooks: Google Colab





