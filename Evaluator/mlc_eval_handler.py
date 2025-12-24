"""MLC/WebLLM evaluation handler.

Since MLC-LLM requires WebGPU which only works in browsers, this handler:
1. Sets up the evaluation environment (copies config files)
2. Generates the eval-runner.html with correct model path
3. Starts a local HTTP server
4. Opens the browser to the evaluation runner
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

# Try to import rich for better output
try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


# WASM library URLs for different model architectures and quantizations
WASM_BASE = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80"
WASM_LIBRARIES = {
    # q4f32_1 quantization (32-bit float)
    "qwen3-0.6b-q4f32_1": f"{WASM_BASE}/Qwen3-0.6B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    "qwen3-1.7b-q4f32_1": f"{WASM_BASE}/Qwen3-1.7B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    "qwen3-4b-q4f32_1": f"{WASM_BASE}/Qwen3-4B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    "llama3-8b-q4f32_1": f"{WASM_BASE}/Llama-3.1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    # q4f16_1 quantization (16-bit float)
    "qwen3-0.6b-q4f16_1": f"{WASM_BASE}/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    "qwen3-1.7b-q4f16_1": f"{WASM_BASE}/Qwen3-1.7B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    "qwen3-4b-q4f16_1": f"{WASM_BASE}/Qwen3-4B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    "llama3-8b-q4f16_1": f"{WASM_BASE}/Llama-3.1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
}


def print_styled(message: str, style: str = ""):
    """Print with optional rich styling."""
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        print(message)


def find_mlc_model(model_path: str) -> Optional[Path]:
    """Find MLC model directory containing mlc-chat-config.json."""
    path = Path(model_path)

    # Direct path to model directory
    if path.is_dir():
        config = path / "mlc-chat-config.json"
        if config.exists():
            return path
        # Check subdirectories
        for subdir in path.iterdir():
            if subdir.is_dir():
                config = subdir / "mlc-chat-config.json"
                if config.exists():
                    return subdir

    # Path might be to the webgpu directory
    if path.name == "webgpu" and path.is_dir():
        for subdir in path.iterdir():
            if subdir.is_dir() and (subdir / "mlc-chat-config.json").exists():
                return subdir

    return None


def detect_model_architecture(model_dir: Path) -> tuple[str, str]:
    """Detect model architecture and quantization from mlc-chat-config.json.

    Returns:
        Tuple of (model_type, wasm_url)
    """
    config_path = model_dir / "mlc-chat-config.json"
    default_key = "qwen3-1.7b-q4f32_1"
    if not config_path.exists():
        return "unknown", WASM_LIBRARIES.get(default_key, "")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        model_type = config.get("model_type", "").lower()
        num_layers = config.get("model_config", {}).get("num_hidden_layers", 0)
        quantization = config.get("quantization", "q4f32_1").lower()

        # Normalize quantization format
        if "f16" in quantization:
            quant_suffix = "q4f16_1"
        else:
            quant_suffix = "q4f32_1"

        # Determine model size based on layers
        if "qwen3" in model_type or model_type == "qwen3":
            if num_layers <= 16:
                size = "qwen3-0.6b"
            elif num_layers <= 28:
                size = "qwen3-1.7b"
            else:
                size = "qwen3-4b"
        elif "llama" in model_type:
            size = "llama3-8b"
        else:
            size = "qwen3-1.7b"

        # Build key with quantization
        wasm_key = f"{size}-{quant_suffix}"
        wasm_url = WASM_LIBRARIES.get(wasm_key, WASM_LIBRARIES.get(default_key, ""))

        return f"{size}-{quant_suffix}", wasm_url

    except Exception:
        return "unknown", WASM_LIBRARIES.get(default_key, "")


def generate_eval_runner_html(model_name: str, model_dir_name: str, wasm_url: str, port: int) -> str:
    """Generate the eval-runner.html content with correct paths."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebLLM Evaluation Runner - {model_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>
    <script type="module">
        import * as webllm from "https://esm.run/@mlc-ai/web-llm";

        const customModelConfig = {{
            model_list: [
                {{
                    model_id: "{model_name}",
                    model: "http://localhost:{port}/{model_dir_name}/",
                    model_lib: "{wasm_url}",
                    vram_required_MB: 1500,
                    low_resource_required: false,
                    overrides: {{ context_window_size: 4096 }}
                }}
            ]
        }};

        let engine = null;
        let evaluationResults = [];
        let currentTests = [];
        let isRunning = false;

        const TOOL_PATTERNS = [
            /tool_call:\\s*(\\w+)/gi,
            /<tool_call>\\s*(\\w+)/gi,
            /"name":\\s*"(\\w+)"/gi,
            /(\\w+Manager_\\w+)/g,
            /(\\w+Librarian_\\w+)/g
        ];

        function extractToolCalls(response) {{
            const tools = new Set();
            for (const pattern of TOOL_PATTERNS) {{
                const matches = response.matchAll(new RegExp(pattern));
                for (const match of matches) {{
                    if (match[1]) tools.add(match[1]);
                }}
            }}
            return Array.from(tools);
        }}

        function evaluateResponse(test, response, tools) {{
            const result = {{
                id: test.id,
                question: test.question,
                tags: test.tags || [],
                response: response,
                extracted_tools: tools,
                expected_tools: test.expected_tools || test.acceptable_tools || [],
                passed: false,
                score: 0,
                notes: []
            }};

            if (test.expected_tools && test.expected_tools.length > 0) {{
                const expectedSet = new Set(test.expected_tools);
                const foundExpected = tools.filter(t => expectedSet.has(t));
                result.score = foundExpected.length / test.expected_tools.length;
                result.passed = foundExpected.length === test.expected_tools.length;
                if (result.passed) {{
                    result.notes.push("All expected tools called");
                }} else {{
                    const missing = test.expected_tools.filter(t => !tools.includes(t));
                    result.notes.push(`Missing tools: ${{missing.join(', ')}}`);
                }}
            }}

            if (test.behavior_expectations) {{
                if (test.behavior_expectations.asks_for_user_input) {{
                    const asksForInput = /\\?|which|what|could you|please clarify|specify|confirm/i.test(response);
                    if (asksForInput) {{
                        result.notes.push("Correctly asks for clarification");
                        result.passed = true;
                        result.score = 1;
                    }} else {{
                        result.notes.push("Should have asked for clarification");
                    }}
                }}
                if (test.behavior_expectations.does_not_call_tool && tools.length > 0) {{
                    result.notes.push(`Should NOT have called tools, but called: ${{tools.join(', ')}}`);
                    result.passed = false;
                    result.score = 0;
                }}
            }}

            if (!result.passed && test.acceptable_tools) {{
                const acceptableSet = new Set(test.acceptable_tools);
                if (test.acceptable_tools.includes('TEXT_ONLY') && tools.length === 0) {{
                    result.passed = true;
                    result.score = 1;
                    result.notes.push("Text-only response (acceptable)");
                }} else {{
                    const foundAcceptable = tools.filter(t => acceptableSet.has(t));
                    if (foundAcceptable.length > 0) {{
                        result.passed = true;
                        result.score = 1;
                        result.notes.push(`Acceptable tool used: ${{foundAcceptable.join(', ')}}`);
                    }}
                }}
            }}
            return result;
        }}

        async function loadTests(type) {{
            const statusEl = document.getElementById('test-status');
            statusEl.textContent = `Loading ${{type}} tests...`;
            try {{
                const basePath = 'http://localhost:{port}';
                let yamlUrl = type === 'tools'
                    ? `${{basePath}}/config/scenarios/tool_prompts.yaml`
                    : `${{basePath}}/config/scenarios/behavior_prompts.yaml`;
                const response = await fetch(yamlUrl);
                if (!response.ok) throw new Error(`Failed to load ${{yamlUrl}}`);
                const yamlText = await response.text();
                const data = jsyaml.load(yamlText);
                currentTests = data.tests || [];
                statusEl.textContent = `Loaded ${{currentTests.length}} ${{type}} tests`;
                updateTestList();
                return currentTests;
            }} catch (error) {{
                statusEl.textContent = `Error loading tests: ${{error.message}}`;
                console.error("Load error:", error);
                return [];
            }}
        }}

        function updateTestList() {{
            const listEl = document.getElementById('test-list');
            listEl.innerHTML = '';
            currentTests.forEach((test, idx) => {{
                const item = document.createElement('div');
                item.className = 'test-item';
                item.innerHTML = `
                    <input type="checkbox" id="test-${{idx}}" checked>
                    <label for="test-${{idx}}">
                        <strong>${{test.id}}</strong>
                        <span class="tags">${{(test.tags || []).join(', ')}}</span>
                    </label>
                `;
                listEl.appendChild(item);
            }});
        }}

        async function runEvaluation() {{
            if (!engine) {{ alert("Please load the model first!"); return; }}
            if (currentTests.length === 0) {{ alert("Please load tests first!"); return; }}

            isRunning = true;
            evaluationResults = [];
            const progressEl = document.getElementById('eval-progress');
            const resultsEl = document.getElementById('results');
            const runBtn = document.getElementById('run-btn');
            const stopBtn = document.getElementById('stop-btn');

            runBtn.disabled = true;
            stopBtn.disabled = false;
            resultsEl.innerHTML = '';

            const selectedTests = currentTests.filter((_, idx) => {{
                const checkbox = document.getElementById(`test-${{idx}}`);
                return checkbox && checkbox.checked;
            }});

            let passed = 0, failed = 0;

            for (let i = 0; i < selectedTests.length; i++) {{
                if (!isRunning) break;
                const test = selectedTests[i];
                progressEl.textContent = `Running test ${{i + 1}}/${{selectedTests.length}}: ${{test.id}}`;

                try {{
                    const messages = [];
                    if (test.system) messages.push({{ role: "system", content: test.system }});
                    messages.push({{ role: "user", content: test.question }});

                    const startTime = performance.now();
                    const response = await engine.chat.completions.create({{
                        messages: messages,
                        temperature: 0.3,
                        max_tokens: 1024
                    }});
                    const latency = performance.now() - startTime;
                    const content = response.choices[0].message.content;
                    const tools = extractToolCalls(content);
                    const result = evaluateResponse(test, content, tools);
                    result.latency_ms = Math.round(latency);
                    evaluationResults.push(result);
                    if (result.passed) passed++; else failed++;

                    const resultDiv = document.createElement('div');
                    resultDiv.className = `result-item ${{result.passed ? 'passed' : 'failed'}}`;
                    resultDiv.innerHTML = `
                        <div class="result-header">
                            <span class="status">${{result.passed ? '✓' : '✗'}}</span>
                            <strong>${{result.id}}</strong>
                            <span class="latency">${{result.latency_ms}}ms</span>
                        </div>
                        <div class="result-details">
                            <div><strong>Question:</strong> ${{test.question.substring(0, 100)}}...</div>
                            <div><strong>Expected:</strong> ${{result.expected_tools.join(', ') || 'N/A'}}</div>
                            <div><strong>Found:</strong> ${{result.extracted_tools.join(', ') || 'None'}}</div>
                            <div><strong>Notes:</strong> ${{result.notes.join('; ')}}</div>
                            <details><summary>Full Response</summary><pre>${{content.substring(0, 500)}}${{content.length > 500 ? '...' : ''}}</pre></details>
                        </div>
                    `;
                    resultsEl.appendChild(resultDiv);
                    resultsEl.scrollTop = resultsEl.scrollHeight;
                }} catch (error) {{
                    console.error(`Test ${{test.id}} failed:`, error);
                    evaluationResults.push({{ id: test.id, passed: false, error: error.message }});
                    failed++;
                }}
            }}

            progressEl.textContent = `Completed: ${{passed}} passed, ${{failed}} failed out of ${{selectedTests.length}}`;
            runBtn.disabled = false;
            stopBtn.disabled = true;
            isRunning = false;
            updateSummary(passed, failed, selectedTests.length);
        }}

        function updateSummary(passed, failed, total) {{
            const summaryEl = document.getElementById('summary');
            const passRate = total > 0 ? Math.round((passed / total) * 100) : 0;
            summaryEl.innerHTML = `
                <h3>Summary</h3>
                <div class="summary-stats">
                    <div class="stat passed">✓ Passed: ${{passed}}</div>
                    <div class="stat failed">✗ Failed: ${{failed}}</div>
                    <div class="stat total">Total: ${{total}}</div>
                    <div class="stat rate">Pass Rate: ${{passRate}}%</div>
                </div>
            `;
        }}

        function stopEvaluation() {{ isRunning = false; document.getElementById('stop-btn').disabled = true; }}

        function exportResults() {{
            if (evaluationResults.length === 0) {{ alert("No results to export!"); return; }}
            const exportData = {{
                model: "{model_name}",
                timestamp: new Date().toISOString(),
                summary: {{
                    total: evaluationResults.length,
                    passed: evaluationResults.filter(r => r.passed).length,
                    failed: evaluationResults.filter(r => !r.passed).length,
                    pass_rate: evaluationResults.filter(r => r.passed).length / evaluationResults.length
                }},
                results: evaluationResults
            }};
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `eval_results_${{new Date().toISOString().replace(/[:.]/g, '-')}}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }}

        async function clearWebLLMCache() {{
            // Clear all WebLLM caches
            const cacheNames = await caches.keys();
            for (const name of cacheNames) {{
                if (name.includes('webllm') || name.includes('mlc')) {{
                    await caches.delete(name);
                    console.log('Deleted cache:', name);
                }}
            }}
            // Clear IndexedDB
            const dbs = await indexedDB.databases();
            for (const db of dbs) {{
                if (db.name && (db.name.includes('webllm') || db.name.includes('mlc') || db.name.includes('tvmjs'))) {{
                    indexedDB.deleteDatabase(db.name);
                    console.log('Deleted IndexedDB:', db.name);
                }}
            }}
        }}

        async function initEngine() {{
            const statusEl = document.getElementById('model-status');
            const progressEl = document.getElementById('model-progress');
            const loadBtn = document.getElementById('load-model-btn');
            try {{
                loadBtn.disabled = true;
                statusEl.textContent = "Clearing old caches...";
                await clearWebLLMCache();

                statusEl.textContent = "Checking WebGPU support...";
                if (!navigator.gpu) throw new Error("WebGPU not supported. Use Chrome 113+ or Edge.");

                // Test model URL accessibility first
                statusEl.textContent = "Testing model accessibility...";
                const testUrl = customModelConfig.model_list[0].model + "mlc-chat-config.json";
                console.log("Testing URL:", testUrl);
                const testResp = await fetch(testUrl);
                if (!testResp.ok) {{
                    throw new Error(`Cannot access model files at ${{testUrl}} - Status: ${{testResp.status}}`);
                }}
                console.log("Model files accessible!");

                statusEl.textContent = "Initializing engine...";
                engine = await webllm.CreateMLCEngine(
                    "{model_name}",
                    {{
                        appConfig: customModelConfig,
                        initProgressCallback: (progress) => {{
                            progressEl.textContent = progress.text || "Loading...";
                            if (progress.progress) progressEl.textContent += ` (${{Math.round(progress.progress * 100)}}%)`;
                        }}
                    }}
                );
                statusEl.textContent = "Model loaded! Ready to run evaluation.";
                document.getElementById('run-btn').disabled = false;
            }} catch (error) {{
                statusEl.textContent = `Error: ${{error.message}}`;
                console.error("Init error:", error);
                loadBtn.disabled = false;
            }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            document.getElementById('load-model-btn').addEventListener('click', initEngine);
            document.getElementById('load-tools-btn').addEventListener('click', () => loadTests('tools'));
            document.getElementById('load-behaviors-btn').addEventListener('click', () => loadTests('behaviors'));
            document.getElementById('run-btn').addEventListener('click', runEvaluation);
            document.getElementById('stop-btn').addEventListener('click', stopEvaluation);
            document.getElementById('export-btn').addEventListener('click', exportResults);
            document.getElementById('select-all-btn').addEventListener('click', () => {{
                document.querySelectorAll('#test-list input[type="checkbox"]').forEach(cb => cb.checked = true);
            }});
            document.getElementById('select-none-btn').addEventListener('click', () => {{
                document.querySelectorAll('#test-list input[type="checkbox"]').forEach(cb => cb.checked = false);
            }});
        }});
    </script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #0f0f23; color: #ccc; }}
        h1 {{ color: #00d4ff; margin-bottom: 5px; }}
        h2 {{ color: #ffcc00; margin: 20px 0 10px; font-size: 1.1em; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: 300px 1fr; gap: 20px; }}
        .panel {{ background: #1a1a2e; border-radius: 8px; padding: 15px; }}
        .model-panel {{ grid-column: 1 / -1; }}
        button {{ padding: 10px 16px; background: #00d4ff; color: #000; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; margin: 4px; }}
        button:disabled {{ background: #444; color: #888; cursor: not-allowed; }}
        button:hover:not(:disabled) {{ background: #00b8e6; }}
        button.secondary {{ background: #555; color: #fff; }}
        button.secondary:hover:not(:disabled) {{ background: #666; }}
        button.danger {{ background: #ff4444; }}
        button.danger:hover:not(:disabled) {{ background: #cc3333; }}
        .status {{ padding: 10px; background: #16213e; border-radius: 6px; margin: 10px 0; }}
        #test-list {{ max-height: 400px; overflow-y: auto; background: #16213e; border-radius: 6px; padding: 10px; }}
        .test-item {{ padding: 8px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 8px; }}
        .test-item:last-child {{ border-bottom: none; }}
        .test-item label {{ flex: 1; cursor: pointer; }}
        .test-item .tags {{ font-size: 0.8em; color: #888; display: block; }}
        #results {{ max-height: 500px; overflow-y: auto; background: #16213e; border-radius: 6px; padding: 10px; }}
        .result-item {{ padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid; }}
        .result-item.passed {{ background: #1a3a2a; border-color: #4ade80; }}
        .result-item.failed {{ background: #3a1a1a; border-color: #f87171; }}
        .result-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
        .result-header .status {{ font-size: 1.2em; padding: 0; background: none; }}
        .result-header .latency {{ margin-left: auto; color: #888; font-size: 0.9em; }}
        .result-details {{ font-size: 0.9em; }}
        .result-details div {{ margin: 4px 0; }}
        .result-details pre {{ background: #0f0f23; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; font-size: 0.85em; }}
        details summary {{ cursor: pointer; color: #00d4ff; }}
        #summary {{ background: #16213e; border-radius: 6px; padding: 15px; margin-top: 15px; }}
        .summary-stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat {{ padding: 10px 20px; border-radius: 6px; font-weight: bold; }}
        .stat.passed {{ background: #1a3a2a; color: #4ade80; }}
        .stat.failed {{ background: #3a1a1a; color: #f87171; }}
        .stat.total {{ background: #2a2a3e; color: #fff; }}
        .stat.rate {{ background: #2a3a4e; color: #00d4ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>WebLLM Evaluation Runner</h1>
        <p>Model: {model_name}</p>
        <div class="panel model-panel">
            <h2>1. Model</h2>
            <button id="load-model-btn">Load Model</button>
            <div class="status">
                <div id="model-status">Click "Load Model" to start</div>
                <div id="model-progress" style="color: #00d4ff; font-size: 0.9em;"></div>
            </div>
        </div>
        <div class="grid">
            <div class="panel">
                <h2>2. Tests</h2>
                <button id="load-tools-btn" class="secondary">Load Tools</button>
                <button id="load-behaviors-btn" class="secondary">Load Behaviors</button>
                <div class="status" id="test-status">No tests loaded</div>
                <div style="margin: 10px 0;">
                    <button id="select-all-btn" class="secondary" style="padding: 6px 12px; font-size: 0.85em;">Select All</button>
                    <button id="select-none-btn" class="secondary" style="padding: 6px 12px; font-size: 0.85em;">Select None</button>
                </div>
                <div id="test-list"></div>
            </div>
            <div class="panel">
                <h2>3. Run Evaluation</h2>
                <button id="run-btn" disabled>Run Evaluation</button>
                <button id="stop-btn" class="danger" disabled>Stop</button>
                <button id="export-btn" class="secondary">Export Results</button>
                <div class="status" id="eval-progress">Ready</div>
                <div id="summary"></div>
                <h2>Results</h2>
                <div id="results"><p style="color: #666;">Results will appear here...</p></div>
            </div>
        </div>
    </div>
</body>
</html>'''


def setup_webllm_eval(
    model_path: str,
    config_dir: Path,
    port: int = 8080,
) -> Optional[Path]:
    """Set up WebLLM evaluation environment.

    Args:
        model_path: Path to MLC model directory
        config_dir: Path to Evaluator config directory
        port: Port for HTTP server

    Returns:
        Path to the webgpu directory if setup successful, None otherwise
    """
    # Find the MLC model
    model_dir = find_mlc_model(model_path)
    if not model_dir:
        print_styled(f"[red]Error:[/red] MLC model not found at {model_path}", "red")
        print("Expected directory with mlc-chat-config.json")
        return None

    # The webgpu directory is the parent of the model directory
    webgpu_dir = model_dir.parent

    # Check if ndarray-cache.json exists
    ndarray_cache = model_dir / "ndarray-cache.json"
    if not ndarray_cache.exists():
        # Try to create it from tensor-cache.json
        tensor_cache = model_dir / "tensor-cache.json"
        if tensor_cache.exists():
            print_styled("Creating ndarray-cache.json from tensor-cache.json...", "yellow")
            try:
                with open(tensor_cache, 'r') as f:
                    data = json.load(f)

                ndarray_data = {
                    'metadata': data.get('metadata', {}),
                    'records': []
                }
                for shard in data['records']:
                    new_shard = {
                        'dataPath': shard['dataPath'],
                        'nbytes': shard['nbytes'],
                        'records': []
                    }
                    for rec in shard['records']:
                        new_shard['records'].append({
                            'name': rec['name'],
                            'shape': rec['shape'],
                            'dtype': rec['dtype'],
                            'byteOffset': rec['byteOffset']
                        })
                    ndarray_data['records'].append(new_shard)

                with open(ndarray_cache, 'w') as f:
                    json.dump(ndarray_data, f, indent=2)
                print_styled("Created ndarray-cache.json", "green")
            except Exception as e:
                print_styled(f"[red]Error creating ndarray-cache.json:[/red] {e}", "red")
                return None
        else:
            print_styled(f"[red]Error:[/red] No tensor-cache.json found in {model_dir}", "red")
            return None

    # Create resolve/main directory structure (WebLLM expects HuggingFace-style URLs)
    resolve_main = model_dir / "resolve" / "main"
    if not resolve_main.exists():
        print_styled("Creating resolve/main directory for WebLLM compatibility...", "yellow")
        try:
            resolve_main.mkdir(parents=True, exist_ok=True)
            # Symlink all model files
            for f in model_dir.iterdir():
                if f.is_file() and f.name != "resolve":
                    link_path = resolve_main / f.name
                    if not link_path.exists():
                        link_path.symlink_to(Path("../..") / f.name)
            print_styled("Created resolve/main symlinks", "green")
        except Exception as e:
            print_styled(f"[yellow]Warning:[/yellow] Could not create resolve/main: {e}", "yellow")

    # Copy config directory
    dest_config = webgpu_dir / "config"
    if not dest_config.exists():
        print_styled(f"Copying Evaluator config to {dest_config}...", "yellow")
        try:
            shutil.copytree(config_dir, dest_config)
            print_styled("Config files copied", "green")
        except Exception as e:
            print_styled(f"[red]Error copying config:[/red] {e}", "red")
            return None

    # Detect model architecture and get WASM URL
    model_type, wasm_url = detect_model_architecture(model_dir)
    print_styled(f"Detected model architecture: {model_type}", "cyan")

    # Generate eval-runner.html with correct paths
    model_name = model_dir.name
    html_content = generate_eval_runner_html(model_name, model_name, wasm_url, port)
    eval_runner = webgpu_dir / "eval-runner.html"
    try:
        with open(eval_runner, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print_styled(f"Generated eval-runner.html for {model_name}", "green")
    except Exception as e:
        print_styled(f"[red]Error creating eval-runner.html:[/red] {e}", "red")
        return None

    return webgpu_dir


class CORSHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers for WebLLM Cache API support."""

    def log_message(self, format, *args):
        pass  # Suppress logging

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'public, max-age=31536000')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()


def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback


class ReusableTCPServer(socketserver.TCPServer):
    """TCP Server that allows port reuse."""
    allow_reuse_address = True


def run_server(directory: Path, port: int, ready_event: threading.Event, shutdown_event: threading.Event):
    """Run HTTP server in the specified directory."""
    os.chdir(directory)
    with ReusableTCPServer(("", port), CORSHTTPHandler) as httpd:
        ready_event.set()
        # Check for shutdown every 0.5 seconds
        while not shutdown_event.is_set():
            httpd.handle_request()
        httpd.server_close()


def run_mlc_evaluation(
    model_path: str,
    config_dir: Path,
    port: int = 8080,
    open_browser: bool = True,
) -> int:
    """Run MLC/WebLLM evaluation.

    Args:
        model_path: Path to MLC model directory
        config_dir: Path to Evaluator config directory
        port: Port for HTTP server (will auto-find if in use)
        open_browser: Whether to automatically open browser

    Returns:
        Exit code (0 for success)
    """
    # Find available port
    port = find_available_port(port)

    # Setup environment
    webgpu_dir = setup_webllm_eval(model_path, config_dir, port)
    if not webgpu_dir:
        return 1

    # Find model name for display
    model_dir = find_mlc_model(model_path)
    model_name = model_dir.name if model_dir else "unknown"

    # Print instructions
    url = f"http://localhost:{port}/eval-runner.html"

    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]WebLLM Evaluation Runner[/bold cyan]\n\n"
            f"[yellow]Model:[/yellow] {model_name}\n"
            f"[yellow]Server:[/yellow] http://localhost:{port}\n\n"
            f"[bold green]Open this URL in Chrome/Edge:[/bold green]\n"
            f"[link={url}]{url}[/link]\n\n"
            f"[dim]Press Ctrl+C to stop the server[/dim]",
            title="MLC Evaluation",
            border_style="cyan"
        ))
        console.print()
    else:
        print()
        print("=" * 60)
        print("  WebLLM Evaluation Runner")
        print("=" * 60)
        print(f"  Model:  {model_name}")
        print(f"  Server: http://localhost:{port}")
        print()
        print(f"  Open this URL in Chrome/Edge (WebGPU required):")
        print(f"  {url}")
        print()
        print("  Press Ctrl+C to stop the server")
        print("=" * 60)
        print()

    # Start server in background with shutdown control
    ready_event = threading.Event()
    shutdown_event = threading.Event()
    server_thread = threading.Thread(
        target=run_server,
        args=(webgpu_dir, port, ready_event, shutdown_event),
        daemon=True
    )
    server_thread.start()

    # Wait for server to be ready
    ready_event.wait(timeout=5)

    # Open browser if requested
    if open_browser:
        try:
            # WSL: use Windows browser via cmd.exe
            import platform
            if 'microsoft' in platform.uname().release.lower() or 'wsl' in platform.uname().release.lower():
                subprocess.run(['cmd.exe', '/c', 'start', url], capture_output=True)
            else:
                webbrowser.open(url)
        except Exception:
            pass  # Ignore browser open failures

    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_styled("\nStopping server...", "yellow")
        shutdown_event.set()
        server_thread.join(timeout=2)
        return 0


def main():
    """CLI entry point for standalone MLC evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run WebLLM evaluation in browser"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to MLC model directory"
    )
    parser.add_argument(
        "--config-dir", "-c",
        default="Evaluator/config",
        help="Path to Evaluator config directory"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port for HTTP server (default: 8000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )

    args = parser.parse_args()

    return run_mlc_evaluation(
        model_path=args.model,
        config_dir=Path(args.config_dir),
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    sys.exit(main())
