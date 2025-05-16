#(C)Tsubasa Kato - Inspire Search Corp. 5/17/2025 - 1:47AM JST
from __future__ import annotations

import itertools
import json
import logging
import random
import re
import uuid
from typing import List, Optional

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from ollama import Client, ResponseError

###############################################################################
# Flask + Ollama bootstrap                                                    #
###############################################################################
app = Flask(__name__)
client = Client(host="http://localhost:11434")  # Ollama daemon must be alive

###############################################################################
# Configuration knobs                                                         #
###############################################################################
MODEL_NAME = "phi4:latest"
GENERATIONS = 3
POP_SIZE = 1
MUTATION_RATE = 0.35

TOPIC_MAP = {
    "automatic": ["AI-driven", "self-optimizing", "adaptive"],
    "programming": ["code generation", "algorithm design", "system architecture"],
    "optimization": ["performance tuning", "resource management", "parallelization"],
}

###############################################################################
# Logging (console)                                                           #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)5s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

###############################################################################
# Genetic algorithm data structure                                            #
###############################################################################
class CodeSpecimen:
    """One piece of Python code + GA metadata."""

    def __init__(self, code: str, parents: Optional[List[str]] | None = None):
        self.code = code or "# <empty>"
        self.parents: List[str] = parents or []
        self.id = str(uuid.uuid4())[:8]
        self.fitness: float = 0.0
        self.evaluate()

    def evaluate(self) -> None:
        score = len(self.code) * 0.10
        score += self.code.count("\n") * 0.50
        if "def " in self.code:
            score += 15
        if "import " in self.code:
            score += 10
        if any(k in self.code.lower() for k in ("ai", "neural", "learning")):
            score += 20
        self.fitness = round(score, 2)

    def __repr__(self) -> str:
        return f"<Specimen {self.id} fit={self.fitness:0.2f}>"

###############################################################################
# Ollama helpers                                                              #
###############################################################################
def ollama_chunks(prompt: str):
    try:
        for chunk in client.generate(model=MODEL_NAME, prompt=prompt, stream=True):
            yield chunk.get("response", "")
    except ResponseError as e:
        yield f"# LLM failed – {e.error}"
    except Exception as e:
        yield f"# Unexpected error – {e}"

###############################################################################
# Prompt helpers                                                              #
###############################################################################
def expand_seeds(base: List[str]) -> List[tuple[str, ...]]:
    expanded: List[List[str]] = []
    for seed in base:
        variations = TOPIC_MAP.get(seed.lower(), []) or [seed]
        expanded.append([seed] + variations)
    return list(itertools.product(*expanded))

def make_prompt(concepts: tuple[str, ...]) -> str:
    return (
        "Create a *complete* Python 3 function that demonstrates "
        + " ".join(concepts)
        + ".\n• The function must have clear inputs and outputs.\n"
        + "• Include error handling and docstrings.\n"
        + "• Return ONLY valid Python code – no markdown fences."
    )

###############################################################################
# Genetic operators                                                           #
###############################################################################
def crossover(a: "CodeSpecimen", b: "CodeSpecimen") -> "CodeSpecimen":
    a_lines = a.code.splitlines()
    b_lines = b.code.splitlines()
    cut = max(1, min(len(a_lines), len(b_lines)) // 2)
    child_code = "\n".join(a_lines[:cut] + b_lines[cut:])
    return CodeSpecimen(child_code, parents=[a.id, b.id])

def mutate(spec: "CodeSpecimen") -> "CodeSpecimen":
    tiny_prompt = (
        "Refactor and improve the following Python 3 code while keeping its "
        "functionality unchanged. Return *code only*.\n" + spec.code
    )
    mutated_code = ""
    for chunk in ollama_chunks(tiny_prompt):
        mutated_code += chunk
    return CodeSpecimen(mutated_code, parents=[spec.id])

###############################################################################
# SSE producer                                                                #
###############################################################################
def _yield_log(msg: str):
    """Utility: yield a log-type SSE event."""
    yield f"event: log\ndata: {msg}\n\n"

def _yield_code(text: str):
    """Utility: yield a code-type SSE event with text content."""
    cleaned = text
    yield f"event: code\ndata: {cleaned}\n\n"

def _yield_specimen_info(specimen: CodeSpecimen, index: int = None, total: int = None, generation: int = None):
    """Utility: yield specimen information as a log event."""
    info = f"Specimen {specimen.id} (fitness: {specimen.fitness}"
    if index is not None and total is not None:
        info += f", {index}/{total}"
    if generation is not None:
        info += f", gen {generation}"
    if specimen.parents:
        info += f", parents: {', '.join(specimen.parents)}"
    info += ")"
    yield from _yield_log(info)

def evolve_stream_generator(seeds: List[str]):
    combos = expand_seeds(seeds)
    population: List[CodeSpecimen] = []

    # ----- initial population -------------------------------------------------
    yield from _yield_log(f"Initialising population ({POP_SIZE}) …")
    for i in range(POP_SIZE):
        prompt = make_prompt(random.choice(combos))
        yield from _yield_log(f"Generating specimen {i+1}/{POP_SIZE}...")

        # Stream the code generation in real-time, chunk by chunk
        code_buffer = ""
        for chunk in ollama_chunks(prompt):
            yield from _yield_code(chunk)  # <<-- Only stream the new chunk!
            code_buffer += chunk

        specimen = CodeSpecimen(code_buffer)
        population.append(specimen)
        yield from _yield_specimen_info(specimen, i+1, POP_SIZE)

    # ----- evolution loop ----------------------------------------------------
    for gen in range(1, GENERATIONS + 1):
        population.sort(key=lambda s: s.fitness, reverse=True)
        yield from _yield_log(f"Generation {gen} | {POP_SIZE} specimens")

        # Display all specimens in the current generation
        for idx, specimen in enumerate(population):
            yield from _yield_specimen_info(specimen, idx+1, POP_SIZE, gen)
            yield from _yield_code(specimen.code)

        # create next generation
        next_pop: List[CodeSpecimen] = [population[0]]  # elitism
        yield from _yield_log(f"Elite specimen {population[0].id} preserved (fitness: {population[0].fitness})")

        while len(next_pop) < POP_SIZE:
            if random.random() < MUTATION_RATE:
                # Mutation with streaming
                parent = random.choice(population)
                yield from _yield_log(f"Mutating specimen {parent.id}...")

                tiny_prompt = (
                    "Refactor and improve the following Python 3 code while keeping its "
                    "functionality unchanged. Return *code only*.\n" + parent.code
                )

                code_buffer = ""
                for chunk in ollama_chunks(tiny_prompt):
                    yield from _yield_code(chunk)  # <<-- Only stream the new chunk!
                    code_buffer += chunk

                child = CodeSpecimen(code_buffer, parents=[parent.id])
            else:
                # Crossover
                parents = random.sample(population, 2)
                yield from _yield_log(f"Crossing over specimens {parents[0].id} and {parents[1].id}...")
                child = crossover(*parents)
                yield from _yield_code(child.code)

            next_pop.append(child)
            yield from _yield_specimen_info(child)

        population = next_pop

    # ----- finished ----------------------------------------------------------
    population.sort(key=lambda s: s.fitness, reverse=True)
    best = population[0]
    yield from _yield_log(f"Evolution complete. Best specimen: {best.id} (fitness: {best.fitness})")
    yield from _yield_code(best.code)

    payload = json.dumps({
        "best_code": best.code,
        "fitness": best.fitness,
        "line_count": best.code.count("\n") + 1,
        "all_specimens": [{"id": s.id, "code": s.code, "fitness": s.fitness} for s in population]
    })
    yield f"event: done\ndata: {payload}\n\n"

###############################################################################
# Flask routes                                                                #
###############################################################################
@app.route("/evolve_stream")
def evolve_stream():
    seeds = [s.strip() for s in request.args.get("seeds", "automatic,programming").split(",") if s.strip()]
    return Response(
        stream_with_context(evolve_stream_generator(seeds)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.route("/evolve", methods=["POST"])
def evolve_json():
    seeds = request.get_json(force=True, silent=True).get("seeds", ["automatic", "programming"])
    best = CodeSpecimen("".join(ollama_chunks(make_prompt(tuple(seeds)))))
    return jsonify(best_code=best.code, fitness=best.fitness, line_count=best.code.count("\n") + 1)

@app.route("/")
def index():
    return render_template("evolve.html")

###############################################################################
# Entrypoint                                                                  #
###############################################################################
if __name__ == "__main__":
    app.run(port=5000, debug=False, threaded=True)
