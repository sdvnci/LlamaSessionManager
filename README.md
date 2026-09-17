# LlamaSessionManager

A simple service for running conversations with local AI models through Ollama.
Handles message history, multi-session pooling, audio transcription, image OCR,
tool dispatch, and session persistence


## What you need

- **Python 3.13+** : [python.org](https://python.org)
- **Ollama** : [ollama.com](https://ollama.com) (runs the AI models)
- **ffmpeg** : `winget install ffmpeg` on Windows, or `brew install ffmpeg` on Mac
- **Docker** (optional, for the server) : [docker.com](https://docker.com)
- **NVIDIA GPU** (recommended for transcription): a modern 8 GB card comfortably runs Whisper `large-v3`

## Install

```bash
git clone <repo-url> sessionmanager
cd sessionmanager

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

Pull a model into Ollama so it has something to run:

```bash
ollama pull tinyllama
```


## Chat (no audio needed)

The library works as a standalone AI chat too:

```python
import asyncio
from messaging._types import SessionConfig
from messaging._messaging import MessageSession


async def chat():
    cfg = SessionConfig(
        model="tinyllama",
        name="my-bot",
        system="You are a helpful assistant. Keep answers short.",
        host="http://localhost:11434",
    )

    async with MessageSession.session(config=cfg) as sess:
        msg = sess.parse_content("What is the capital of France?")
        await sess.add_message(msg)
        reply = await sess.send()
        print(reply.message.content)

        # Follow-up : history is preserved automatically
        msg2 = sess.parse_content("And what about Germany?")
        await sess.add_message(msg2)
        reply2 = await sess.send()
        print(reply2.message.content)


asyncio.run(chat())
```

---

## Adding context : files, images, and documents

`parse_content` is how you hand the AI anything beyond plain text.
Pass a string (your instruction) together with one or more file paths,
and it figures out how to read each one.

```python
# One file
msg = sess.parse_content("Summarise this report.", Path("reports/sales.pdf"))

# Multiple files at once
msg = sess.parse_content([
    "Compare the figures in the spreadsheet with what the report says.",
    Path("finance/q3.csv"),
    Path("finance/q3-report.pdf"),
])

await sess.add_message(msg)
reply = await sess.send()
print(reply.message.content)
```

### What file types work

| What you drop in | Extensions | Needs a special model? |
|------------------|------------|------------------------|
| Images | `.png` `.jpg` `.jpeg` `.gif` `.bmp` `.webp` | Yes : a vision model |
| PDFs | `.pdf` | No : any model works |
| Word docs | `.docx` `.docm` `.dotx` `.dotm` | No : any model works |
| Text / code | `.txt` `.md` `.csv` `.json` `.py` `.js` `.yaml` … | No : any model works |
| Spreadsheets | `.csv` (read as raw text) | No : any model works |

### Ask about an image

You need a **vision model** : one that can "see" pictures:

```python
msg = sess.parse_content([
    "Transcribe all the text you see in this image. Output only the text.",
    Path("media/screenshot.png"),
])
await sess.add_message(msg)
reply = await sess.send(think=False)  # think=False skips reasoning, faster
print(reply.message.content)
```

Or from the command line:

```bash
python main.py --mode ocr --source "media/screenshot.png"
```

### Ask about a PDF

```python
msg = sess.parse_content([
    "Summarise this report in three bullet points.",
    Path("reports/q4-sales.pdf"),
])
await sess.add_message(msg)
reply = await sess.send()
```

### Ask about a Word document

```python
msg = sess.parse_content([
    "Who are the signatories on this contract?",
    Path("contracts/nda.docx"),
])
await sess.add_message(msg)
reply = await sess.send()
```

### Mix a photo with typed text

```python
msg = sess.parse_content([
    "Is the address on this invoice the same as the one below?\n"
    "Expected: 123 Main St, Springfield, IL 62701",
    Path("invoices/invoice-2025.png"),
])
await sess.add_message(msg)
reply = await sess.send(think=False)
```

### Follow up after a transcription

Once `main.py` finishes transcribing, the transcript is on disk.
Load it into any chat session to ask follow-ups:

```python
from pathlib import Path
from messaging._messaging import MessageSession

msg = MessageSession.build_transcript_context_message(
    transcript_path=Path("artifacts/.../transcript.txt"),
    instruction="What action items were discussed in this meeting?",
)
await sess.add_message(msg)
reply = await sess.send()
```

### Tips

- **Vision models are slower.** Use `think=False` for OCR and image questions
  to skip reasoning and get faster answers.
- **Long PDFs may not fit.** If the AI misses details, ask about specific
  sections instead of "summarise the whole thing".
- **CSV files are sent as raw text.** The AI sees the comma-separated rows.
  For number crunching, ask it to extract specific columns or add up totals.
- **Images go to the model as-is.** No resizing needed : Ollama handles
  base-64 encoding internally.

---

## Use tools / function calling

Let the AI run your own Python functions:

```python
def get_weather(city: str) -> str:
    return f"Sunny, 22 deg C in {city}"

parsed = await sess.send_with_tools({"get_weather": get_weather})
print(parsed.content)
# "The weather in Paris is Sunny, 22 deg C."
```

---
