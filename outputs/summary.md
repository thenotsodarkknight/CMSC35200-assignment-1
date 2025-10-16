# AI Workflow Summary (English)

First, I re-authenticated with Globus using the Sophia-tools repo and created a four-model "telephone" chain on the Argonne ALCF Sophia service (`task1.py`), logging latencies and outputs for ten prompts. I then rebuilt Open WebUI (`task2.py`), scripted its environment, and verified the Sophia connection by launching chat sessions directly inside the UI.

The gene-analysis automation (`task3.py`) samples fifty HGNC symbols, queries Meta-Llama‑3.1‑70B on Sophia, and emits structured evidence for four disease classes in 364 s, versus ~200 minutes of manual research. I confirmed JSON fidelity by cross-checking each gene entry and flagging sentences with “no known association” for follow-up. Local Ollama runs of Llama 3.2 3B and Phi‑3 3.8B (`task4.py`) replay the pipeline, letting me quantify offline speed/quality trade-offs. Finally, I fine-tuned nanoGPT on Shakespeare with Apple MPS (`task5.py`), captured the training transcript, and plotted the learning curve below.

| Task | Highlights | Key Metrics |
| --- | --- | --- |
| Telephone chain | Four-model paraphrase relay, 10 prompts | Avg latencies: 2.24 → 4.03 → 3.22 → 6.80 |
| Open WebUI | Automated reinstall + Sophia provider | `.env.openwebui` + `OPEN_WEBUI_CONFIG_DIR` bootstrap; in-UI chat verified |
| Gene analysis | Automated disease tagging & QA hooks | 11 cancer, 1 heart-disease associations; 364 s runtime |
| Local models | Ollama replay of gene pipeline | Llama3.2:3b ≈ 106 s; Phi3:3.8b ≈ 183 s |
| nanoGPT | Shakespeare fine-tune on MPS | Val loss dips to 3.59 (step 200);|

# Resumen del Flujo de Trabajo (Español)

Durante este ciclo implementé cinco flujos de trabajo y validé cada uno personalmente antes de pedir ayuda a la IA para pulir el texto. Primero renové la autenticación con Globus y construí una cadena de “teléfono” con cuatro modelos de Sophia (`task1.py`), registrando latencias y salidas de diez prompts. Después reinstalé Open WebUI (`task2.py`), configuré el entorno y comprobé dentro de la interfaz que los chats con Sophia funcionaran, a pesar de que el servicio no expone `/models`.

La automatización de análisis génico (`task3.py`) selecciona cincuenta símbolos HGNC, consulta Meta-Llama‑3.1‑70B y entrega evidencias estructuradas para cuatro clases de enfermedades en 364 s, frente a unas 200 min de revisión manual. Verifiqué cada gen y marqué las frases con “sin asociación conocida” como banderas amarillas. Las ejecuciones locales con Ollama (Llama 3.2 3B y Phi‑3 3.8B, `task4.py`) repiten el flujo y permiten medir velocidad y calidad sin conexión. Por último, el ajuste fino de nanoGPT sobre Shakespeare se ejecuta en la GPU Apple MPS (`task5.py`), y la curva de aprendizaje resultante se muestra abajo.

| Tarea | Aspectos destacados | Métricas clave |
| --- | --- | --- |
| Cadena telefónica | Relevo de paráfrasis con cuatro modelos, 10 prompts | Latencia media (s): 2.24 → 4.03 → 3.22 → 6.80 |
| Open WebUI | Reinstalación automatizada + proveedor Sophia | `.env.openwebui` + `OPEN_WEBUI_CONFIG_DIR`; chats comprobados |
| Análisis génico | Etiquetado automático con controles | 11 cáncer, 1 cardiopatía; ejecución de 364 s |
| Modelos locales | Repetición con Ollama | Llama3.2:3b ≈ 106 s; Phi3:3.8b ≈ 183 s |
| nanoGPT | Ajuste fino en MPS | Val loss mínima 3.59 (paso 200); |

# 100-Word Observation on Tasks A & B

Pending
