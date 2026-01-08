# Theoretical Foundations

This page covers the formal type-theoretic framework underlying ContextFS and type-safe context engineering.

## Formal Framework

### Definitions

!!! note "Definition: Context Type"
    A **context type** $\Gamma$ is a specification that constrains the space of valid responses. Formally, $\Gamma$ defines a set of constraints $\{c_1, c_2, ..., c_n\}$ that any valid response must satisfy.

!!! note "Definition: Response Term"
    A **response term** $t$ is a model output. We write $t : \Gamma$ to indicate that $t$ satisfies all constraints in $\Gamma$.

!!! note "Definition: Type Safety"
    A context $\Gamma$ is **type-safe** if there exists a unique equivalence class of responses $[t]$ such that for all valid responses $t_1, t_2 : \Gamma$, we have $t_1 \sim t_2$ under semantic equivalence.

### The Typing Judgment

We define a typing judgment for context engineering:

$$
\Gamma \vdash t : T
$$

Read as: "Under context $\Gamma$, response $t$ has type $T$."

### Type Constructors

Building complex types from simpler ones:

| Constructor | Meaning | Example |
|-------------|---------|---------|
| $A \times B$ | Product (and) | JSON with fields A and B |
| $A + B$ | Sum (or) | Either format A or B |
| $A \to B$ | Function | Given input A, produce B |
| $\Pi_{x:A} B(x)$ | Dependent | Output type depends on input |

## Dependent Types in Context

The most powerful aspect: **response types can depend on context values**.

```python
# Type depends on the code being reviewed
def review_type(code: str) -> Type[Review]:
    if "async" in code:
        return AsyncCodeReview  # Includes concurrency checks
    else:
        return SyncCodeReview   # Standard checks
```

Formally:

$$
\Pi_{\text{code} : \text{String}} \text{ReviewType}(\text{code})
$$

## The Correspondence

| Programming Languages | Context Engineering |
|----------------------|---------------------|
| Source code | Context (prompt + examples) |
| Type specification | Output schema |
| Compilation | Inference |
| Type error | Invalid response |
| Type inference | Understanding implicit constraints |
| Subtyping | Response satisfies stricter constraints |

## Proof-Theoretic View

Under Curry-Howard correspondence:

- **Types** = Propositions
- **Terms** = Proofs
- **Type inhabitation** = Provability

Context engineering becomes **proof search**:

Given context $\Gamma$ (propositions/assumptions), find response $t$ (proof) such that $t : T$ (proof of proposition $T$).

### Why AlphaFold Succeeds

Protein folding is proof search where:

- Sequence = Type constraints
- Structure = Proof term
- Anfinsen's dogma = Type safety guarantee

AlphaFold learned efficient proof search heuristics for a well-typed problem.

## Failure Modes as Type Errors

### Underdetermined Types

$$
\Gamma \vdash t_1 : T \quad \Gamma \vdash t_2 : T \quad t_1 \not\sim t_2
$$

Multiple non-equivalent valid responses exist.

**Example:**
```
Context: "Write a function"
Valid responses: Any function in any language
```

### Overdetermined Types (Uninhabited)

$$
\nexists t. \; \Gamma \vdash t : T
$$

No valid response can satisfy all constraints.

**Example:**
```
Context: "Write a 10-word essay covering all of machine learning in depth"
```

### Type Mismatch

$$
\Gamma \vdash t : T' \quad T' \not<: T
$$

Response satisfies a different type than expected.

**Example:**
```
Expected: JSON object
Got: Markdown text
```

## Chaperone Systems

Inspired by molecular chaperones that guide protein folding:

### Retry Chaperone

```python
def retry_chaperone(prompt, target_type, max_attempts=3):
    for attempt in range(max_attempts):
        response = model.generate(prompt)
        try:
            return target_type.model_validate_json(response)
        except ValidationError as e:
            prompt = f"{prompt}\n\nPrevious attempt failed: {e}\nTry again:"
    raise TypingError("Could not inhabit type")
```

### Progressive Refinement Chaperone

```python
def progressive_chaperone(task, target_type):
    # Start with broad type
    outline = model.generate(f"Outline: {task}")

    # Refine to specific type
    details = model.generate(
        f"Given outline:\n{outline}\n\nNow provide: {target_type.schema()}"
    )

    return target_type.model_validate_json(details)
```

### Ensemble Chaperone

```python
def ensemble_chaperone(prompt, target_type, n=3):
    responses = [model.generate(prompt) for _ in range(n)]
    validated = [target_type.model_validate_json(r) for r in responses]
    return consensus(validated)  # Aggregate consistent responses
```

## Implications for AI Safety

### Alignment as Type Safety

If we could specify human values as a type $V$, alignment becomes:

$$
\forall \text{action} \; a. \; \Gamma_{\text{values}} \vdash a : V
$$

All actions must inhabit the "aligned" type.

### Robustness as Type Preservation

System is robust if type safety holds under perturbation:

$$
\Gamma \vdash t : T \implies \Gamma' \vdash t' : T \quad \text{for small } \delta(\Gamma, \Gamma')
$$

### Interpretability as Type Inference

Understanding a model's behavior = inferring what type constraints it has learned:

$$
\text{Given outputs } t_1, t_2, ..., t_n \text{, infer } \Gamma \text{ such that } \Gamma \vdash t_i : T
$$

## Practical Applications

### 1. Prompt Engineering Guidelines

From type theory:

- **Be explicit**: Specify output types precisely
- **Be complete**: Ensure constraints determine unique response class
- **Be consistent**: Avoid contradictory constraints

### 2. Evaluation Metrics

- **Type inhabitation rate**: % of responses that validate against schema
- **Semantic consistency**: Agreement between responses to same prompt
- **Constraint satisfaction**: % of explicit constraints met

### 3. System Design

- Use Pydantic/JSON Schema for explicit typing
- Implement chaperone patterns for reliability
- Design for progressive type refinement

## ContextFS Implementation

ContextFS implements these theoretical foundations through a formal type system based on **Definition 5.1 Type Grammar**:

```
MemoryType    ::= Mem Schema           -- Schema-indexed memory
VersionedType ::= Versioned MemoryType -- Timeline with ChangeReason
```

### Mem[S] - Type-Safe Memory

```python
from contextfs.types import Mem
from contextfs.schemas import DecisionData

# Type-safe access to structured_data
typed: Mem[DecisionData] = memory.as_typed(DecisionData)
print(typed.data.decision)  # IDE knows this is str
```

### VersionedMem[S] - Evolution Tracking

Memory evolution follows four formal change reasons:

| ChangeReason | Type-Theoretic Meaning |
|--------------|------------------------|
| `OBSERVATION` | New axiom added to context |
| `INFERENCE` | Derived theorem from premises |
| `CORRECTION` | Contradiction resolution |
| `DECAY` | Axiom confidence reduction |

```python
from contextfs.types import VersionedMem, ChangeReason

versioned = memory.as_versioned(DecisionData)
versioned.evolve(new_data, reason=ChangeReason.CORRECTION)
```

For implementation details, see [Formal Type System](formal-type-system.md).

## References

1. Anfinsen, C. B. (1973). Principles that govern the folding of protein chains. *Science*.

2. Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*.

3. Wadler, P. (2015). Propositions as Types. *Communications of the ACM*.

4. Martin-Löf, P. (1984). *Intuitionistic Type Theory*.

5. Long, M. & YonedaAI Collaboration. (2025). Type-Safe Context Engineering. *Preprint*.

6. Long, M. (2025). [Typed Memory Paper](../paper/typed-memory-paper.pdf). Definition 5.1 Type Grammar.
