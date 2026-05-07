## AMD MI300X vLLM Server Setup

On your AMD Developer Cloud instance, start the OpenAI-compatible vLLM server with:

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --device cuda --tensor-parallel-size 1 --max-model-len 8192
```
