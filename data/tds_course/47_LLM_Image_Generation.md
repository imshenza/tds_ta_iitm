Gemini Flash Experimental Image Generation and Editing APIs
In March 2025, Google introduced native image generation and editing capabilities in the Gemini 2.0 Flash Experimental model. You can now generate and iteratively edit images via a single REST endpoint (
Experiment with Gemini 2.0 Flash native image generation
, 
Generate images | Gemini API | Google AI for Developers
).
 (
How to use Latest Gemini 2.0 Native Image Generation with API?
)
Simple image generation
To generate a basic image, send a POST request to the 
generateContent
 method:
curl
 
"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key=
$GEMINI_API_KEY
"
 
\

  
-H
 
"Content-Type: application/json"
 
\

  
-X
 POST 
\

  
-d
 
'{
    "contents": [{ "parts": [{ "text": "A serene landscape of rolling hills at sunrise, digital art" }] }],
    "generationConfig": { "responseModalities": ["TEXT", "IMAGE"] }
  }'
 
|
 jq 
-r
 
'.candidates[].content.parts[] | select(.inlineData) | .inlineData.data'
 
|
 base64 
--decode
 
>
 image.png
Copy to clipboard
Error
Copied
Replace 
$GEMINI_API_KEY
 with your key. (
Gemini API | Google AI for Developers
)
Generation options
You can tweak the output with these 
generationConfig
 parameters:
responseModalities
: Modalities to return (
TEXT
, 
IMAGE
).
temperature
 (0.0–2.0): Controls randomness (default 1.0).
topP
 (0.0–1.0): Nucleus sampling threshold.
topK
: Token selection cutoff.
maxOutputTokens
: Max tokens for text parts.
stopSequences
: Sequences to end generation.
seed
: For reproducibility.
curl
 
"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key=
$GEMINI_API_KEY
"
 
\

  
-H
 
"Content-Type: application/json"
 
\

  
-X
 POST 
\

  
-d
 
'{
    "contents": [{ "parts": [{ "text": "A futuristic city skyline at dusk, neon lights" }] }],
    "generationConfig": {
      "responseModalities": ["TEXT", "IMAGE"],
      "temperature": 0.7,
      "topP": 0.9,
      "maxOutputTokens": 1024
    }
  }'
 
|
 jq 
-r
 
'.candidates[].content.parts[] | select(.inlineData) | .inlineData.data'
 
|
 base64 
--decode
 
>
 image.png
Copy to clipboard
Error
Copied
Image Generation Docs
Simple image editing
To edit an existing image, include it in the 
contents
 as 
inlineData
 (base64-encoded):
curl
 
"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key=
$GEMINI_API_KEY
"
 
\

    
-H
 
'Content-Type: application/json'
 
\

    
-d
 
'{
      "contents": [{
        "parts":[
            {"text": "Replace the background with a starry night sky"},
            {"inline_data": {"mime_type":"image/jpeg", "data": "'
$(
base64 
-w
 
0
 cat.jpg
)
'"}}
        ]
      }],
      "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    }'
 
|
 jq 
-r
 
'.candidates[].content.parts[] | select(.inlineData) | .inlineData.data'
 
|
 base64 
--decode
 
>
 image.png
Copy to clipboard
Error
Copied
Image Editing Docs
Editing options
Editing requests support:
inlineData
: Embed raw image bytes.
fileData
: Reference public URLs.
All 
generationConfig
 options listed above.
safetySettings
: Per-request safety rules.
Multi-turn edits by repeating 
contents
 in conversation history.
Costs and optimization
Gemini 2.0 Flash Experimental uses token-based billing:
Input
 (text/image/video): free tier, then $0.10 per 1M tokens.
Output
 (text/image): free tier, then $0.40 per 1M tokens.
Per-image flat cost
 for Pro models: ~$0.001315 /image (
Gemini Developer API Pricing | Gemini API | Google AI for Developers
, 
Solved: Re: Outdated Gemini Pro image pricing? By tile, or…
).
To optimize:
Use smaller image sizes by setting 
responseMimeType
.
Cache or reuse prompts with 
cachedContent
.
Lower 
candidateCount
 or 
temperature
 for fewer tokens.
OpenAI gpt-image-1 Model for Image Generation and Editing
OpenAI’s GPT Image 1 (
gpt-image-1
) is a state-of-the-art multimodal model released on April 23, 2025, for high-fidelity image creation and editing.
Simple image generation
Use the Image Generations endpoint:
curl
 
'https://api.openai.com/v1/images/generations'
 
\

  
-H
 
'Content-Type: application/json'
 
\

  
-H
 
"Authorization: Bearer 
$OPENAI_API_KEY
"
 
\

  
-d
 
'{
    "model": "gpt-image-1",
    "prompt": "A whimsical illustration of a cat playing chess",
    "n": 1,
    "size": "1024x1024"
  }'
 
>
 image.png
Copy to clipboard
Error
Copied
(
Generate Image | OpenAI API - Postman
)
Generation options
Adjust these JSON parameters:
model
: 
gpt-image-1
 (default).
prompt
: Text description.
n
: Number of images.
size
: 
256x256
, 
512x512
, or 
1024x1024
.
response_format
: 
"url"
 (default) or 
"b64_json"
.
{

  
"model"
:
 
"gpt-image-1"
,

  
"prompt"
:
 
"..."
,

  
"n"
:
 
2
,

  
"size"
:
 
"512x512"
,

  
"response_format"
:
 
"b64_json"


}
Copy to clipboard
Error
Copied
Simple image editing
Use the Edits endpoint with an image and a mask:
curl
 https://api.openai.com/v1/images/edits 
\

  
-H
 
'Content-Type: application/json'
 
\

  
-H
 
"Authorization: Bearer 
$OPENAI_API_KEY
"
 
\

  
-d
 
'{
    "model": "gpt-image-1",
    "image": "data:image/png;base64,<BASE64_IMAGE>",
    "mask": "data:image/png;base64,<BASE64_MASK>",
    "prompt": "Add a rainbow in the sky above the mountains",
    "n": 1,
    "size": "1024x1024"
  }'
Copy to clipboard
Error
Copied
(
curl - What’s the correct URL to test OpenAI API? - Stack Overflow
)
Editing options
Editing requests accept:
image
: Original image (base64 or URL).
mask
: PNG mask for inpainting.
prompt
: Instruction for the edit.
n
, 
size
, 
response_format
 as above.
Optional 
user
 field for attribution.
Costs and optimization
GPT Image 1 pricing (per 1M tokens): text input $5, image input $10, image output $40. Rough per-image costs:
Low quality: ~$0.02
Medium quality: ~$0.07
High quality: ~$0.19 (
OpenAI’s GPT-Image-1 API — Create Stunning Images for Your Apps!
, 
Usage of gpt-image-1 is priced per token, with … - Hacker News
)
To optimize:
Choose smaller sizes (
256x256
).
Generate fewer images (
n:1
).
Use 
response_format:"url"
 to reduce payload.
Cache frequent prompts or images.














Previous




LLM Agents












Next










LLM Speech





