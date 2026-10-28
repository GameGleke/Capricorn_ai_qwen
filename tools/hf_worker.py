"""Воркер инференса модели с Hugging Face для Капри.

Запускается отдельным процессом из hf_backend.py (sys.executable), общается JSON-строками:
запросы в stdin, ответы в stdout. Так тяжёлый torch/CUDA живёт вне Qt-процесса и не вешает аватар.

    python tools/hf_worker.py Qwen/Qwen3.5-9B

Протокол (одна JSON-строка на запрос, поле id возвращается в ответе):
  {"cmd":"load"}                                           -> {"id":1,"ok":true}
  {"cmd":"generate","messages":[...],"max_new_tokens":300} -> {"id":2,"text":"..."}
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


def _device(torch):
    if torch.cuda.is_available():
        return {"device_map": "auto"}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"device_map": "mps"}
    return {}  # CPU: медленнее, но работает; веса остаются в RAM (~18 ГБ bf16)


def _tf_version():
    import transformers
    parts = transformers.__version__.split(".")
    return tuple(int(x) for x in parts[:2] if x.isdigit())


def _arch_supported(src):
    """Сверка architectures из config.json модели с классами установленного transformers.
    Без этой проверки Капри молча уходила в поиск: у новой модели неизвестную архитектру
    AutoModel не поднимает, а воркер умирал без объяснений."""
    try:
        cfg_path = os.path.join(src, "config.json") if os.path.isdir(src) else \
            __import__("huggingface_hub").hf_hub_download(src, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            archs = json.load(f).get("architectures", [])
    except Exception:
        return None  # не удалось проверить — считаем, что поддерживается
    try:
        import transformers
        known = set(dir(transformers))
        return all(a in known for a in archs) if archs else None
    except Exception:
        return None


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # локальная папка с весами (если скачана вручную в hf_models/) — иначе общий кэш Hugging Face
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(root, "hf_models", model_id.replace("/", "--"))
    src = local if os.path.isfile(os.path.join(local, "config.json")) else model_id
    tfv = ".".join(map(str, _tf_version()))
    if _arch_supported(src) is False:
        raise RuntimeError(f"transformers {tfv} не знает модель {model_id} — обнови: "
                           'python -m pip install -U "transformers>=4.57"')
    tok = AutoTokenizer.from_pretrained(src)
    kwargs = {"dtype": torch.bfloat16} if _tf_version() >= (4, 56) else {"torch_dtype": torch.bfloat16}
    try:
        model = AutoModelForCausalLM.from_pretrained(src, **kwargs, **_device(torch))
    except Exception as e:
        msg = str(e)
        if "no matching model type" in msg or "does not recognize" in msg or "architecture" in msg.lower():
            raise RuntimeError(f"transformers {tfv} не поддерживает {model_id}: обновись до версии >= 4.57 "
                               '(python -m pip install -U "transformers>=4.57")') from e
        raise
    model.eval()
    return tok, model


# хвосты чата и рассуждения «мысли вслух» выкидываем из финального текста
STOP_TAIL = re.compile(r"<\|im_end\|>.*|<\|endoftext\|>.*|</think>.*", re.S)


def generate(tok, model, messages, max_new_tokens):
    import torch
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)  # Qwen3: не рассуждать вслух
    except (TypeError, ValueError):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.6, top_p=0.9, repetition_penalty=1.15,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    truncated = out[0].shape[0] - inputs["input_ids"].shape[1] >= max_new_tokens
    text = STOP_TAIL.sub("", text).strip()
    if truncated:
        text += "[truncated]"  # упёрлась в лимит токенов — hf_backend почистит хвост
    return text


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
    tok = model = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id")
        cmd = req.get("cmd")
        try:
            if cmd == "load":
                tok, model = load_model(model_id)
                send({"id": rid, "ok": True})
            elif cmd == "generate":
                if model is None:
                    raise RuntimeError("модель не загружена")
                text = generate(tok, model, req["messages"], int(req.get("max_new_tokens", 300)))
                send({"id": rid, "text": text})
            elif cmd == "ping":
                send({"id": rid, "ok": model is not None})
            else:
                send({"id": rid, "error": f"неизвестная команда {cmd}"})
        except Exception as e:
            err(e)
            send({"id": rid, "error": str(e)[:400]})


if __name__ == "__main__":
    main()
