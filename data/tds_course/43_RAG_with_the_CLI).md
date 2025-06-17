Retrieval Augmented Generation (RAG) with the CLI
Retrieval Augmented Generation (RAG) combines retrieval (searching a knowledge base) with generation (using an LLM) to produce answers grounded in your own documents. Instead of relying solely on a general-purpose LLM, RAG lets you feed it the most relevant chunks from your corpus at query time, improving accuracy, reducing hallucinations, and allowing you to answer domain‑specific questions without fine‑tuning.
In particular, you can answer questions that are hard to answer with a keyword search. For example:
Q
=
"What does the author affectionately call the => syntax?"


# Answer: fat arrow



Q
=
"What lets you walk every child node of a ts.Node?"


# Answer: node.getChildren()



Q
=
"What are code pieces like comments and whitespace that aren’t in the AST called?"


# Answer: trivia



Q
=
"Which operator converts any value into an explicit boolean?"


# Answer: !!
Copy to clipboard
Error
Copied
You can implement RAG entirely from your terminal, without writing a single line of application code. Below is a step‑by‑step example using the TypeScript book as a data source.
1. Clone the repository
git
 clone 
--depth
 
1
 https://github.com/basarat/typescript-book

cd
 typescript-book
Copy to clipboard
Error
Copied
--depth 1
 fetches only the latest commit to minimize download size.
cd typescript-book
 moves into the project folder.
You’ll now be in a new folder 
typescript-book
 containing the repo.
2. Split Markdown files into chunks
(

  
shopt
 
-s
 globstar
  
for
 
f
 
in
 **/*.md
;
 
do

    uvx 
--from
 split_markdown4gpt mdsplit4gpt 
"
$f
"
 
--model
 gpt-4o 
--limit
 
4096
 
--separator
 
"===SPLIT==="
 
\

    
|
 
sed
 
'1s/^/===SPLIT===\n/'
 
\

    
|
 jq 
-R
 
-s
 
-c
 
--arg
 
file
 
"
$f
"
 
'
      split("===SPLIT===")[1:]
      | to_entries
      | map({
          id: ($file + "#" + (.key | tostring)),
          content: .value
        })[]
    '

  
done


)
 
|
 
tee
 chunks.json
Copy to clipboard
Error
Copied
shopt -s globstar
: lets 
**/*.md
 match Markdown files in all subdirectories. 
bash shopt manual
uvx --from split_markdown4gpt mdsplit4gpt
: 
a tool
 that splits Markdown into LLM‑sized chunks:
--model gpt-4o
: uses GPT‑4o token limits
--limit 4096
: max tokens per chunk
--separator "===SPLIT==="
: custom split marker
sed '1s/^/===SPLIT===\n/'
: ensures the very first chunk starts with the separator (GNU sed manual)
jq -R -s -c --arg file "$f"
: uses 
jq
 to convert chunks to JSON
-R
: read raw input
-s
: slurp entire input into a single string
-c
: compact JSON output
builds an array of objects 
{id, content}
, where 
id
 is 
filename#chunkIndex
tee chunks.json
: writes the resulting JSON array to 
chunks.json
 while printing it to stdout.
You’ll now have a 
chunks.json
 that has one 
{id, content}
 JSON object per line.
3. Generate embeddings
llm embed-multi typescript-book 
--model
 
3
-small 
--store
 
--format
 
nl
 chunks.json
Copy to clipboard
Error
Copied
embed-multi
: computes embeddings for each entry in 
chunks.json
.
typescript-book
: a namespace or collection name for storage.
--model 3-small
: selects the embedding model.
--store
: save embeddings in the default backend.
--format nl
: input is newline‑delimited JSON. 
llm CLI embed-multi
This stores the embeddings in a collection called 
typescript-book
.
llm collections path  
# shows where the collections are stored

llm collections delete typescript-book  
# deletes the typescript-book collection
Copy to clipboard
Error
Copied
4. Find similar topics
llm similar typescript-book 
-n
 
3
 
-c
 
"What does the author affectionately call the => syntax?"
Copy to clipboard
Error
Copied
This returns the 3 chunksmost similar to the question posed.
similar
: retrieves the top 
n
 most similar chunks from the embeddings store.
-n 3
: return three results.
-c
: the user’s query string.
5. Answer a question using retrieved context
Q
=
"What does the author affectionately call the => syntax?"

llm similar typescript-book 
-n
 
3
 
-c
 
"
$Q
"
 
\

  
|
 jq 
'.content'
 
\

  
|
 llm 
-s
 
"
$Q
 - Answer ONLY from these notes. Cite verbatim from notes."
 
\

  
|
 uvx streamdown
Copy to clipboard
Error
Copied
This answers the question in natural language following these steps:
Store the query in 
Q
.
Retrieve the top 3 matching chunks.
jq '.content'
 extracts just the text snippets.
Pipe into 
llm -s
, instructing the model:
-s
: stream a prompt directly to the LLM.
"$Q - Answer ONLY from these notes. Cite verbatim from notes."
: ensures the response is grounded.
uvx streamdown
 formats the streamed LLM output for easy reading.
















Previous




Vector databases












Next










Hybrid RAG with TypeSense





