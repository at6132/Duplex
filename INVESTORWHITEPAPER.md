# Duplex Neural Hot‑Swap Architecture

## A Cognitive Control Plane for Foundation Models

---

# Executive Overview

Modern AI models are extraordinarily powerful — but they are fundamentally static once trained.

If something changes in the world:

* A new law is passed
* A policy updates
* A constraint changes mid-task
* A user corrects an assumption
* A live data feed delivers new information

The model cannot truly *revise itself* in motion.

It must stop.
Restart.
Reprocess everything.

This is not how intelligent systems should work.

Duplex introduces a new architectural layer that allows foundation models to maintain a **persistent, revisable internal state** — enabling real-time correction, redirection, and control without retraining or restarting.

We call this:

> A Neural Hot‑Swap Module.

It transforms static models into dynamically steerable reasoning engines.

---

# The Core Breakthrough (Explained Simply)

Today's models operate like this:

1. You give them input.
2. They generate output.
3. If something changes, you must stop and start over.

Internally, reasoning and expression are fused together. Once the model starts speaking, its internal reasoning is tightly bound to the text it has already produced. New information has nowhere to go — it cannot reach the model's reasoning without restarting the entire sequence.

Duplex separates two things that were previously fused:

* **Reasoning state** (what the model "believes" right now)
* **Expression** (what the model is currently saying)

Instead of thinking only through emitted tokens, the model maintains a structured internal workspace — a live cognitive state that can be updated independently of the text stream. New information is encoded directly into this latent state, not appended to a growing input sequence.

The workspace conditions future output.
When the workspace changes, the model's reasoning changes — mid-sentence, mid-task, in real time.

The result:

* The model can change course mid-generation.
* Corrections modify internal belief, not prompt text.
* New constraints alter the reasoning pathway without a full reset.

This is a shift from:

> Static token continuation

To:

> Persistent, revisable cognition.

---

# What Makes This Different From Existing Methods

This is not:

* A larger model
* A fine-tune of any kind
* A prompt engineering technique
* LoRA or any weight-modification adapter
* Retrieval-augmented generation (RAG)

Here is precisely why:

**RAG** retrieves documents and places them into the input context before generation begins. It is a pre-generation, text-level operation. It grows the context window. It cannot update the model once generation has started. It is reactive — triggered per query, paid for per query.

**LoRA and adapters** modify or extend the model's weights. The change is permanent, always active, and baked in at training time. You cannot swap a LoRA adapter mid-inference. You cannot update it without another training run. It is an offline operation that produces a static artifact.

**Fine-tuning and continued pretraining** rewrite the model's parameters entirely. New knowledge overwrites old. It is expensive, permanent, and happens between deployments — never during them.

All three approaches share the same fundamental constraint:

> They solve what the model knows **before it starts talking.**

Duplex solves an entirely different problem:

> What happens to the model's reasoning when new information arrives **after it starts talking.**

That gap — mid-generation belief revision — is what no existing method addresses. It is the only problem Duplex is designed to solve, and it solves it at the architecture level, not the prompt level.

---

# Why This Matters

## 1. Real-Time Correction Without Restart

Today:

* Model generates
* User interrupts
* System restarts from scratch — context lost, time wasted, continuity broken

With Duplex:

* Model generates
* User corrects
* Workspace state updates via gated cross-attention
* Generation continues immediately, redirected — no stop, no restart, no lost context

The correction does not enter as text appended to a prompt. It is encoded into the latent workspace and conditions the model's next token directly through prefix attention. The distinction matters: this is belief revision, not context extension.

This enables coding assistants, tutoring systems, collaborative writing, and any workflow where human redirection is part of the process — without the current penalty of starting over every time.

---

## 2. Runtime Reasoning Updates — Without Touching Weights

Large models cost enormous resources to retrain. Existing approaches — LoRA, fine-tuning, continued pretraining — all operate offline, between deployments. They modify weights. The change is permanent, always active, and cannot be undone without another training run. Different contexts cannot receive different reasoning behaviors from the same deployed model.

Duplex operates on an entirely different axis:

* The backbone weights never change — not during training, not during deployment, not ever
* Reasoning behavior is controlled through the workspace state, which is dynamic and ephemeral
* Different conversations carry different workspace states — the same frozen model reasons differently based on what its workspace currently holds
* Workspace states can be updated, replaced, or reset at any point during inference

This is not a weight modification. It is not a retrieval operation. It is a live reasoning state that exists independently of both the model's parameters and its input sequence — and can be changed while the model is mid-sentence.

Instead of rewriting intelligence, we steer it.
Instead of tattooing new information onto the model, we hand it a whiteboard that can be erased and rewritten at any moment.

---

## 3. Hot-Swappable Specialist Modules — Zero Retraining

Current domain specialization requires separate fine-tuned model variants. A hospital with 500 department policies needs 500 models — 500 training runs, 500 deployment artifacts, 500 maintenance burdens. Adding a new domain means a new training run.

Duplex's workspace architecture enables a fundamentally different paradigm.

Each domain is represented as a **trained workspace initialization** — a `(32 × 2048)` latent tensor, approximately 200K parameters, trained independently on domain-specific data. This is not a weight delta. It is a starting cognitive state — a pre-loaded reasoning posture that the model adopts before a single token is generated. The backbone is untouched.

At inference, a lightweight router selects the appropriate workspace initialization based on the incoming request. The dynamic gated update then adapts further from that starting state based on the live conversation.

New domains are added by training a new workspace tensor — an afternoon of compute, no changes to the backbone, no redeployment of the base model.

**Why this is categorically different from Mixture of Experts:**

MoE routing and all its expert sub-networks are frozen into the model at training time. Adding a new expert to a deployed MoE model requires retraining the entire system. Duplex specialist workspaces are independent portable artifacts — train one, register it, load it at runtime. The model never changes. The specialists are infinitely extensible post-deployment.

This creates modular, versionable cognitive layers that compound in value as new domains are added — with zero marginal cost to the backbone.

---

## 4. AI Agents With Native Mid-Task Redirection

Agents today operate in brittle loops:

Observe → Plan → Act → Restart

When something changes mid-task — a new constraint, a corrected assumption, an updated objective — current agents must either ignore it or abort entirely. This is not a scaffolding problem that better tooling can fix. It is a model-level problem: the agent's reasoning state is fused to the token sequence. The token sequence cannot be revised mid-flight. There is no mechanism for the model to absorb a new constraint and continue without reprocessing everything.

With a persistent workspace:

* The agent maintains a live latent belief state about its environment and objectives
* New constraints update the workspace state directly during execution — not appended to a growing prompt
* The agent's subsequent reasoning is conditioned on the updated state without discarding completed work
* Human oversight becomes genuinely real-time — an operator can redirect a running agent mid-task at the reasoning level, not just by sending a new message and hoping

This is the only architectural approach that makes AI agents truly interruptible at the model level. Everything else is scaffolding that works around the fundamental constraint. Duplex removes the constraint.

The AI agent market is projected to reach $47.1B by 2030. Every serious agent deployment has this problem. None have a model-level solution.

---

## 5. Real-Time Streaming Data Ingestion During Generation

Every existing approach to giving a model current information shares the same constraint: the information must be present before generation begins.

RAG retrieves and prepends to the input — a pre-generation operation that grows the context window with every update and cannot run mid-sentence. Longer context windows increase per-token latency proportionally. Re-prompting requires a full restart. None of these can condition the model's output on data that arrives after the first token.

Duplex removes this constraint through a property unique to the workspace architecture: the workspace is a **fixed-size latent state**. It does not grow when new information arrives. It transforms — the same 32 slots hold an updated representation after each gated cross-attention pass. Absorbing a new data point costs a single encoder forward pass and a workspace update. It does not extend the sequence. It does not increase attention complexity. It does not require stopping generation.

This makes continuous data ingestion during generation computationally feasible for the first time.

**What this enables:**

* Live trading analysis where market data updates the model's reasoning state token by token — not retrieved once at prompt time, but continuously absorbed as prices move
* Medical monitoring where patient vitals update the model's diagnostic reasoning in real time during report generation
* Autonomous systems that absorb continuous sensor feeds without replanning cycles
* News summarization that stays current mid-paragraph as the wire updates

**A note on ambient world-awareness for everyday conversational AI:**

Current chatbots handle questions about recent events by triggering a web search — a per-query retrieval operation that costs tokens, adds latency, and scales linearly with usage. An alternative model: run a continuous background process that streams live information into a shared workspace state around the clock. When a user asks about a current event, the workspace already holds a compressed latent representation of the current world state — no search triggered, no tokens spent, no latency added. One continuous update pipeline serves all users simultaneously.

This is distinct from RAG in a precise way: RAG pulls document text into the input context on demand, growing the sequence and paying per query. This is a persistent latent world-state maintained in the workspace — present before any question is asked, updated continuously, and accessed through the model's own prefix attention rather than through context text. The workspace cannot store precise facts the way a retrieved document can, but it carries a compressed signal of current world state — enough to give the model genuine ambient awareness rather than a hard knowledge cutoff.

---

## 6. Video Understanding and Temporal Coherence

Current multimodal systems process video by reasoning over discrete frames, often rebuilding their scene representation from scratch at each step. Temporal continuity is fragile. Object identity drifts across cuts. Long-range coherence degrades.

With a persistent workspace updated frame by frame through sequential gated cross-attention passes:

* Scene state accumulates as a continuous latent representation rather than being rebuilt at each frame
* Object and actor identity persists across cuts and camera changes through the workspace's carried state
* The model's understanding of the scene evolves without reprocessing prior frames
* Multi-shot coherence improves because the workspace carries forward what was understood before

The workspace becomes a dynamic scene canvas — a running latent model of the visual world that evolves continuously rather than resetting between frames.

---

## 7. Robotics and Physical AI

Humanoid and autonomous systems require persistent world models, continuous sensory updates, mid-action correction, and multi-modal integration. Today most systems handle environmental changes by replanning from scratch — computationally expensive, latency-sensitive, and brittle under rapid change.

A workspace layer changes this dynamic:

* The system's belief state about its environment is maintained in the workspace and updated continuously from sensor data — cameras, LiDAR, force feedback — through the same gated cross-attention mechanism
* When something in the environment changes, the workspace updates and the model's next action is conditioned on the revised belief state — without discarding the prior plan entirely
* Mid-action corrections from operators update the workspace state directly during execution — not by sending a new prompt and restarting the planning loop

This moves physical AI systems from stateless per-step prediction toward stateful world reasoning — absorbing continuous reality rather than snapshotting it.

---

# The Architectural Shift

Foundation models today are:

* Massive static priors
* Token continuation engines
* Context-window bound
* Restart-based when new information arrives

Duplex introduces:

* A persistent latent cognitive state that exists independently of both weights and context
* Gated cross-attention update pathways that absorb new information without growing the sequence
* Prefix conditioning that leverages the backbone's own attention — no layer injection, no cascade corruption
* Separation of reasoning state from expression
* Hot-swappable domain workspaces extensible post-deployment with zero backbone changes

This is not incremental scaling.
It is a structural change in where model reasoning lives and how it updates.

---

# What We Have Demonstrated

At small scale, we have already shown:

* A model with an explicit workspace update mechanism revises downstream generation after mid-stream interruption — without stopping, restarting, or extending the context
* Disabling the workspace update path collapses revision accuracy from 55% to 4.4% and raises contradiction rate from 3.6% to 40.2% — confirming the workspace causally controls reasoning, not just surface output
* Prefix conditioning achieves 100% redirect accuracy where cross-attention adapter injection into decoder layers achieved 0% — validating the architectural approach across four design iterations
* The backbone's weights are untouched throughout — every result is produced by a fully frozen Qwen3-1.7B conditioned on a trained 250M parameter workspace layer

We are now scaling this architecture onto larger open-weight models.

---

# The Long-Term Vision

We envision a new primitive in the AI stack:

```
Foundation Model  (frozen, commodity, swappable)
+
Cognitive Control Plane  (workspace layer, dynamic, persistent)
```

This control plane enables:

* Live belief revision during generation — from humans, agents, or data streams
* Modular domain specialization with zero backbone retraining
* Persistent reasoning state across modalities — text, vision, sensor, audio
* Ambient world-awareness maintained continuously rather than retrieved on demand

The core mechanism remains the same across every application:

> New information updates the workspace state.
> The workspace state conditions generation.
> Generation continues — uninterrupted, redirected, current.

And critically — when the next frontier model is released, Duplex does not retrain. The workspace layer installs onto the new backbone. The capability compounds as base models improve. The moat deepens with every new open-weight release, not just with Duplex's own training runs.

This moves AI systems from:

> Stateless prediction engines that know what they were trained on

Toward:

> Stateful reasoning systems that know what is happening right now.

---

# Closing Summary

Duplex is not a better chatbot.
It is not a retrieval system.
It is not a fine-tuning method.

It is an attempt to introduce a missing primitive into foundation models:

> A persistent, revisable internal cognitive state that updates in real time — independent of weights, independent of context, alive during generation.

If successful, this enables a new class of AI systems that are:

* Interruptible mid-generation without restarting
* Updatable without modifying weights
* Modular across domains without retraining
* State-aware across time and modalities
* Continuously current without per-query retrieval

It transforms how intelligence is maintained, redirected, and evolved after deployment.

Without retraining the core model.
Without growing the context window.
Without stopping generation.

That is the breakthrough.