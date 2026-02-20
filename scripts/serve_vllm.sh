#!/bin/bash
#
# vLLM Server Başlatma Scripti — Qwen3-8B-AWQ
#
# Bu script vLLM'i ayrı bir process olarak serve eder.
# OpenAI-compatible API endpoint oluşturur (http://localhost:6365/v1)
#
# Kullanım:
#   ./scripts/serve_vllm.sh                    # Varsayılan ayarlarla
#   ./scripts/serve_vllm.sh --port 6367       # Farklı port
#   ./scripts/serve_vllm.sh --context 4096    # Daha kısa context
#
# Durdurma: Ctrl+C veya kill <PID>

set -e

# ── Varsayılan Ayarlar ────────────────────────────────────────
MODEL="Qwen/Qwen3-8B-AWQ"
PORT="${VLLM_PORT:-6365}"
CONTEXT_LEN="${VLLM_CONTEXT_LEN:-4096}"
GPU_MEMORY="${VLLM_GPU_MEMORY:-0.90}"

# ── Argüman Parsing ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --context)
            CONTEXT_LEN="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT          Server port (default: 6365)"
            echo "  --context LEN        Max context length (default: 4096)"
            echo "  --gpu-memory RATIO   GPU memory utilization (default: 0.90)"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  VLLM_PORT            Server port"
            echo "  VLLM_CONTEXT_LEN     Max context length"
            echo "  VLLM_GPU_MEMORY      GPU memory utilization"
            echo ""
            echo "Note: Qwen3 reasoning mode varsayılan olarak aktif."
            echo "      Kapatmak için API çağrılarında enable_thinking=False kullanın."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ── vLLM Kontrolü ──────────────────────────────────────────────
if ! command -v vllm &> /dev/null; then
    echo "ERROR: vLLM bulunamadi. Kurulum:"
    echo "  pip install vllm>=0.8.5"
    exit 1
fi

# ── CUDA Kontrolü ──────────────────────────────────────────────
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "WARNING: CUDA bulunamadi. CPU mode'da calisacak (cok yavas)."
fi

# ── Model Kontrolü ────────────────────────────────────────────
echo "Model kontrol ediliyor: $MODEL"
if ! python -c "from huggingface_hub import model_info; model_info('$MODEL')" 2>/dev/null; then
    echo "WARNING: Model HuggingFace'de bulunamadi. İlk çalıştırmada indirilecek."
fi

# ── vLLM Server Başlatma ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  vLLM Server Başlatılıyor — Qwen3-8B-AWQ"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "Context:      $CONTEXT_LEN tokens"
echo "GPU Memory:   ${GPU_MEMORY}"
echo "Prefix Cache: Enabled"
echo "Tool Calling: Enabled (hermes parser)"
echo ""
echo "API Endpoint: http://localhost:$PORT/v1"
echo "OpenAI Compatible: Yes"
echo ""
echo "Durdurmak için: Ctrl+C"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# vLLM komutunu oluştur
VLLM_CMD="vllm serve $MODEL \
    --port $PORT \
    --quantization awq_marlin \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len $CONTEXT_LEN \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prefix-caching"

# Server'ı başlat
exec $VLLM_CMD
