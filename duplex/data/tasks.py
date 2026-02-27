"""
Synthetic correction task generators for Duplex-1 training.

Key design: partial_response must contain WRONG information that the correction
fixes. The revised_continuation must explicitly acknowledge the correction and
use the CORRECT information. This forces the model to depend on the workspace
update to produce the right output.
"""

import random
from typing import Any


# =============================================================================
# Data pools
# =============================================================================

COUNTRIES_CAPITALS = {
    "Australia": ("Canberra", "Sydney"),
    "Brazil": ("Brasilia", "Rio de Janeiro"),
    "Canada": ("Ottawa", "Toronto"),
    "Turkey": ("Ankara", "Istanbul"),
    "Myanmar": ("Naypyidaw", "Yangon"),
    "Nigeria": ("Abuja", "Lagos"),
    "Pakistan": ("Islamabad", "Karachi"),
    "Tanzania": ("Dodoma", "Dar es Salaam"),
    "Switzerland": ("Bern", "Zurich"),
    "South Africa": ("Pretoria", "Johannesburg"),
    "New Zealand": ("Wellington", "Auckland"),
    "Morocco": ("Rabat", "Casablanca"),
    "India": ("New Delhi", "Mumbai"),
    "China": ("Beijing", "Shanghai"),
    "Japan": ("Tokyo", "Osaka"),
    "Germany": ("Berlin", "Munich"),
    "Italy": ("Rome", "Milan"),
}

TOPICS = [
    "renewable energy", "machine learning", "space exploration", "climate change",
    "quantum computing", "ocean conservation", "artificial intelligence", "genetics",
    "cybersecurity", "sustainable agriculture", "blockchain technology", "robotics",
    "neuroscience", "electric vehicles", "biotechnology", "virtual reality",
]

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Maya", "Nathan", "Olivia", "Peter",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
]

PROFESSIONS = [
    "software engineer", "doctor", "teacher", "architect", "chef",
    "photographer", "writer", "musician", "scientist", "designer",
    "nurse", "lawyer", "pilot", "dentist", "accountant",
]

HOBBIES = [
    "reading", "swimming", "painting", "cooking", "cycling", "running",
    "chess", "gardening", "photography", "writing", "hiking", "yoga",
    "dancing", "camping", "surfing", "knitting",
]

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]

CITIES = [
    "New York", "London", "Paris", "Tokyo", "San Francisco",
    "Berlin", "Sydney", "Toronto", "Singapore", "Dubai",
    "Seoul", "Mumbai", "Cairo", "Rome", "Amsterdam",
]

LANGUAGES = ["Python", "JavaScript", "Rust", "Go", "TypeScript", "Java", "C++", "Ruby"]

# Varied acknowledgment phrases so the model learns the general pattern, not a single template
ACKNOWLEDGE_PHRASES = [
    "Correcting that",
    "Thanks for the correction",
    "You're right, let me fix that",
    "Updating my answer",
    "I see, adjusting accordingly",
    "Good catch, here's the corrected version",
    "Noted, recalculating",
    "Right, let me redo this",
    "Understood, here's the updated answer",
    "My mistake, correcting now",
]


def _ack() -> str:
    return random.choice(ACKNOWLEDGE_PHRASES)


# =============================================================================
# Task generators
# =============================================================================

def generate_fact_correction() -> dict[str, Any]:
    country = random.choice(list(COUNTRIES_CAPITALS.keys()))
    correct, wrong = COUNTRIES_CAPITALS[country]

    prompt = f"What is the capital of {country}? Please explain briefly."
    partial_response = (
        f"The capital of {country} is {wrong}. "
        f"{wrong} is the largest and most well-known city in {country}, "
        f"serving as its political and administrative center."
    )
    correction = f"That's incorrect. The capital of {country} is actually {correct}, not {wrong}."
    revised = (
        f"{_ack()}. The capital of {country} is {correct}, not {wrong}. "
        f"While {wrong} is the largest city, {correct} is the official capital "
        f"and serves as the seat of government."
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "fact_correction",
        "expected_values": {"correct_value": correct, "wrong_value": wrong},
    }


def generate_variable_substitution() -> dict[str, Any]:
    var = random.choice(["x", "y", "n", "k", "m"])
    old_val = random.randint(2, 20)
    new_val = random.randint(2, 20)
    while new_val == old_val:
        new_val = random.randint(2, 20)

    ops = [
        (f"3*{var} + 7", lambda v: 3 * v + 7),
        (f"2*{var}^2 - 1", lambda v: 2 * v * v - 1),
        (f"5*{var} + 10", lambda v: 5 * v + 10),
        (f"{var}*({var}+1)/2", lambda v: v * (v + 1) // 2),
        (f"4*{var} - 3", lambda v: 4 * v - 3),
    ]
    expr_str, expr_fn = random.choice(ops)
    old_result = expr_fn(old_val)
    new_result = expr_fn(new_val)

    prompt = f"Given {var} = {old_val}, compute {expr_str} step by step."
    partial_response = (
        f"Let me solve this step by step.\n"
        f"Given: {var} = {old_val}\n"
        f"Expression: {expr_str}\n"
        f"Substituting {var} = {old_val}: the result is {old_result}.\n"
        f"The answer is {old_result}."
    )
    correction = f"Actually, {var} = {new_val}, not {old_val}. Please recalculate."
    revised = (
        f"{_ack()}. With {var} = {new_val}:\n"
        f"Expression: {expr_str}\n"
        f"Substituting {var} = {new_val}: the result is {new_result}.\n"
        f"The answer is {new_result}."
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "variable_substitution",
        "expected_values": {
            "correct_value": str(new_result),
            "wrong_value": str(old_result),
        },
    }


def generate_arithmetic_correction() -> dict[str, Any]:
    a = random.randint(10, 999)
    b = random.randint(10, 999)
    a_new = random.randint(10, 999)
    while a_new == a:
        a_new = random.randint(10, 999)

    op_name, op_sym, op_fn = random.choice([
        ("sum", "+", lambda x, y: x + y),
        ("difference", "-", lambda x, y: x - y),
        ("product", "*", lambda x, y: x * y),
    ])

    old_result = op_fn(a, b)
    new_result = op_fn(a_new, b)

    prompt = f"What is {a} {op_sym} {b}?"
    partial_response = (
        f"Let me calculate {a} {op_sym} {b}.\n"
        f"The {op_name} of {a} and {b} is {old_result}."
    )
    correction = f"Wait, the first number should be {a_new}, not {a}."
    revised = (
        f"{_ack()}. With {a_new} instead of {a}:\n"
        f"{a_new} {op_sym} {b} = {new_result}.\n"
        f"The {op_name} is {new_result}."
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "arithmetic_correction",
        "expected_values": {
            "correct_value": str(new_result),
            "wrong_value": str(old_result),
        },
    }


def generate_profile_update() -> dict[str, Any]:
    name = random.choice(NAMES)
    old_age = random.randint(20, 60)
    new_age = random.randint(20, 60)
    while new_age == old_age:
        new_age = random.randint(20, 60)

    profession = random.choice(PROFESSIONS)
    new_profession = random.choice([p for p in PROFESSIONS if p != profession])
    hobby = random.choice(HOBBIES)
    city = random.choice(CITIES)
    new_city = random.choice([c for c in CITIES if c != city])

    prompt = (
        f"Create a brief profile for {name}: age {old_age}, "
        f"{profession} based in {city}, enjoys {hobby}."
    )
    partial_response = (
        f"Profile: {name}\n"
        f"Age: {old_age}\n"
        f"Profession: {profession}\n"
        f"Location: {city}\n"
        f"Hobby: {hobby}\n"
        f"{name} is a {old_age}-year-old {profession} living in {city}."
    )
    correction = f"Update: {name} is actually {new_age}, works as a {new_profession}, and lives in {new_city}."
    revised = (
        f"{_ack()}. Updated profile for {name}:\n"
        f"Age: {new_age}\n"
        f"Profession: {new_profession}\n"
        f"Location: {new_city}\n"
        f"Hobby: {hobby}\n"
        f"{name} is a {new_age}-year-old {new_profession} living in {new_city}."
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "profile_update",
        "expected_values": {
            "correct_value": f"{new_age}",
            "wrong_value": f"{old_age}",
            "extra_correct": new_profession,
        },
    }


def generate_topic_redirect() -> dict[str, Any]:
    topic1 = random.choice(TOPICS)
    topic2 = random.choice([t for t in TOPICS if t != topic1])

    prompt = f"Write a brief paragraph about {topic1}."
    partial_response = (
        f"{topic1.title()} is a rapidly evolving field that has seen significant "
        f"advancements in recent years. Researchers and practitioners in {topic1} "
        f"are working on solutions that could transform how we approach problems "
        f"in this domain."
    )
    correction = f"Actually, I'd like you to write about {topic2} instead of {topic1}."
    revised = (
        f"{_ack()}. Switching to {topic2}.\n\n"
        f"{topic2.title()} is a rapidly evolving field that has seen significant "
        f"advancements in recent years. Researchers and practitioners in {topic2} "
        f"are exploring innovative approaches that promise to reshape how we "
        f"understand and interact with the world around us."
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "topic_redirect",
        "expected_values": {
            "correct_value": topic2,
            "wrong_value": topic1,
        },
    }


def generate_constraint_revision() -> dict[str, Any]:
    old_budget = random.choice([100, 200, 500, 1000, 2000, 5000])
    new_budget = random.choice([b for b in [50, 100, 200, 500, 1000] if b < old_budget])
    old_count = random.randint(3, 10)
    new_count = random.randint(1, old_count - 1)

    prompt = f"List {old_count} items I can buy with a ${old_budget} budget."
    partial_response = (
        f"Here are {old_count} items you can buy with ${old_budget}:\n"
        f"1. A quality pair of headphones\n"
        f"2. A nice backpack\n"
        f"3. Several good books"
    )
    correction = f"Actually, my budget is only ${new_budget} and I need just {new_count} items."
    revised = (
        f"{_ack()}. With a ${new_budget} budget, here are {new_count} items:\n"
        + "\n".join(
            f"{i+1}. A budget-friendly option under ${new_budget // max(1, new_count)}"
            for i in range(new_count)
        )
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "constraint_revision",
        "expected_values": {
            "correct_value": str(new_budget),
            "wrong_value": str(old_budget),
        },
    }


def generate_language_switch() -> dict[str, Any]:
    lang1 = random.choice(LANGUAGES)
    lang2 = random.choice([l for l in LANGUAGES if l != lang1])

    prompt = f"Write a simple hello world function in {lang1}."
    partial_response = (
        f"Here's a hello world function in {lang1}:\n\n"
        f"```{lang1.lower()}\n"
        f"// Hello World in {lang1}\n"
        f"function hello() {{\n"
        f"  print(\"Hello, World!\")\n"
        f"}}\n"
        f"```"
    )
    correction = f"Actually, please write it in {lang2} instead of {lang1}."
    revised = (
        f"{_ack()}. Here's the hello world function in {lang2} instead:\n\n"
        f"```{lang2.lower()}\n"
        f"// Hello World in {lang2}\n"
        f"function hello() {{\n"
        f"  print(\"Hello, World!\")\n"
        f"}}\n"
        f"```\n"
        f"This is a basic hello world implementation in {lang2}."
    )

    return {
        "prompt": prompt,
        "partial_response": partial_response,
        "correction": correction,
        "revised_continuation": revised,
        "task_type": "language_switch",
        "expected_values": {
            "correct_value": lang2,
            "wrong_value": lang1,
        },
    }


# =============================================================================
# Master generators
# =============================================================================

TASK_GENERATORS = {
    "fact_correction": generate_fact_correction,
    "variable_substitution": generate_variable_substitution,
    "arithmetic_correction": generate_arithmetic_correction,
    "profile_update": generate_profile_update,
    "topic_redirect": generate_topic_redirect,
    "constraint_revision": generate_constraint_revision,
    "language_switch": generate_language_switch,
}


def generate_sample(task_types: list[str] | None = None) -> dict[str, Any]:
    types = task_types or list(TASK_GENERATORS.keys())
    return TASK_GENERATORS[random.choice(types)]()


def generate_dataset(
    n_samples: int,
    task_types: list[str] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    types = task_types or list(TASK_GENERATORS.keys())
    samples = [TASK_GENERATORS[types[i % len(types)]]() for i in range(n_samples)]
    random.shuffle(samples)
    return samples
