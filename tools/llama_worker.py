"""Воркер инференса GGUF-модели через llama.cpp (llama-cpp-python) для Капри.

Запускается отдельным процессом из llama_backend.py, общается JSON-строками:
запросы в stdin, ответы в stdout — тот же протокол, что у hf_worker.py.

    python tools/llama_worker.py unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf

Все ошибки печатаются в stderr отдельными строками "ERROR: ..." — их читает
llama_backend и пишет в capri.log / показывает пользователю. Раньше воркер
умирал молча (stderr был закрыт в DEVNULL), и Капри на всё отвечала «загугли».
"""
import json
import os
import re
import sys


def err(text):
    print("ERROR: " + str(text)[:300], file=sys.stderr, flush=True)


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def find_local(fname):
    """Локально скачанный .gguf (папки gguf_models/ или кэш huggingface-cli) — чтобы не качать заново."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    direct = os.path.join(root, "gguf_models", fname)
    if os.path.isfile(direct):
        return direct
    places = [os.path.join(root, "hf_cache", "hub"),
              os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    for hub in places:
        if not os.path.isdir(hub):
            continue
        for d in os.listdir(hub):
            snaps = os.path.join(hub, d, "snapshots")
            if not os.path.isdir(snaps):
                continue
            for snap in os.listdir(snaps):
                f = os.path.join(snaps, snap, fname)
                if os.path.isfile(f):
                    return f
    return None


def load_model(repo_id, fname):
    from llama_cpp import Llama

    path = find_local(fname)
    if not path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("нет библиотеки huggingface_hub — запусти setup_llama.bat")
        # если HF перекрыт сетью, можно задать зеркало: set HF_ENDPOINT=https://hf-mirror.com
        path = hf_hub_download(repo_id, fname)

    ctx = int(os.environ.get("CAPRI_LLAMA_CTX", "8192"))
    gpu = int(os.environ.get("CAPRI_LLAMA_GPU_LAYERS", "99"))  # нехватку VRAM llama.cpp сам частично скинет на CPU
    try:
        llm = Llama(model_path=path, n_ctx=ctx, n_gpu_layers=gpu, verbose=False)
    except TypeError:  # старые сборки без части аргументов
        llm = Llama(model_path=path, n_ctx=ctx)
    return llm


# хвосты чата и рассуждения «мысли вслух» выкидываем из финального текста
STOP_TAIL = re.compile(r"<\|im_end\|>.*|<\|endoftext\|>.*|</think>.*", re.S)


def generate(llm, messages, max_new_tokens):
    kwargs = dict(max_tokens=max_new_tokens, temperature=0.6, top_p=0.9, repeat_penalty=1.15)
    if not any(m.get("role") == "system" for m in messages):
        kwargs["add_generation_prompt"] = True  # голый user-запрос всё равно должен открыть блок ассистента
    try:
        out = llm.create_chat_completion(messages=messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):  # шаблон/версия без переключателя размышлений
        out = llm.create_chat_completion(messages=messages, **kwargs)
    text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    truncated = out.get("finish_reason") == "length"
    text = STOP_TAIL.sub("", text).strip()
    if truncated:
        text += "[truncated]"
    return text


def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "unsloth/Qwen3.5-9B-GGUF"
    fname = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.5-9B-Q4_K_M.gguf"
    llm = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid, cmd = req.get("id"), req.get("cmd")
        try:
            if cmd == "load":
                llm = load_model(repo_id, fname)
                send({"id": rid, "ok": True})
            elif cmd == "generate":
                if llm is None:
                    raise RuntimeError("модель не загружена")
                text = generate(llm, req["messages"], int(req.get("max_new_tokens", 300)))
                send({"id": rid, "text": text})
            elif cmd == "ping":
                send({"id": rid, "ok": llm is not None})
            else:
                send({"id": rid, "error": "неизвестная команда %s" % cmd})
        except Exception as e:
            err(e)
            send({"id": rid, "error": str(e)[:400]})


if __name__ == "__main__":
    main()
