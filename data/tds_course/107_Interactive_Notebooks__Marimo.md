Interactive Notebooks: Marimo
Marimo
 is a new take on notebooks that solves some headaches of Jupyter. It runs cells reactively - when you change one cell, all dependent cells update automatically, just like a spreadsheet.
Marimo’s cells can’t be run out of order. This makes Marimo more reproducible and easier to debug, but requires a mental shift from the Jupyter/Colab way of working.
It also runs Python directly in the browser and is quite interactive. 
Browse the gallery of examples
. With a wide variety of interactive widgets, It’s growing popular as an alternative to Streamlit for building data science web apps.
Common Operations:
# Create new notebook

uvx marimo new


# Run notebook server

uvx marimo edit notebook
.
py


# Export to HTML

uvx marimo export notebook
.
py
Copy to clipboard
Error
Copied
Best Practices:
Cell Dependencies
Keep cells focused and atomic
Use clear variable names
Document data flow between cells
Interactive Elements
# Add interactive widgets

slider 
=
 mo
.
ui
.
slider
(
1
,
 
100
)


# Create dynamic Markdown

mo
.
md
(
f"
{
slider
}
 {"
🟢
" * slider.value}"
)
Copy to clipboard
Error
Copied
Version Control
Keep notebooks are Python files
Use Git to track changes
Publish on 
marimo.app
 for collaboration














Previous




Narratives with LLMs












Next










HTML Slides: RevealJS





