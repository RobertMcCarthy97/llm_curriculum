
To generate prompt:

```bash
python llm_curriculum/prompts/generate_prompt.py
# Result is saved to llm_curriculum/prompts/txt/prompt.txt
```

Put the resulting prompt into GPT
Example done here: https://chat.openai.com/share/c1630805-520f-43d0-82d7-e6a4925aa049
Get the response, save in `llm_curriculum/prompts/txt/response.txt`

Parse the response and print nicely:
```bash
python llm_curriculum/prompts/parse_response.py
```