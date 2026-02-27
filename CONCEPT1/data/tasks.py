"""
Synthetic task generators for full-duplex interruption/revision experiments.

Each generator produces structured samples with:
    - prompt: initial task description
    - output_prefix: beginning of the model's response (before interruption)
    - update: correction/new information arriving mid-generation
    - revised_continuation: correct continuation after incorporating the update
    - task_type: identifier for the task category
    - expected_values: dict of key facts the post-update output must contain
"""

import random
from typing import Any


# ---------------------------------------------------------------------------
# Shared data pools
# ---------------------------------------------------------------------------

COUNTRIES_CAPITALS = {
    "Australia": "Canberra",
    "Brazil": "Brasilia",
    "Canada": "Ottawa",
    "Turkey": "Ankara",
    "Myanmar": "Naypyidaw",
    "Nigeria": "Abuja",
    "Pakistan": "Islamabad",
    "Tanzania": "Dodoma",
    "Switzerland": "Bern",
    "South Africa": "Pretoria",
    "New Zealand": "Wellington",
    "Morocco": "Rabat",
    "Sri Lanka": "Sri Jayawardenepura Kotte",
    "India": "New Delhi",
    "China": "Beijing",
}

WRONG_CAPITALS = {
    "Australia": "Sydney",
    "Brazil": "Rio de Janeiro",
    "Canada": "Toronto",
    "Turkey": "Istanbul",
    "Myanmar": "Yangon",
    "Nigeria": "Lagos",
    "Pakistan": "Karachi",
    "Tanzania": "Dar es Salaam",
    "Switzerland": "Zurich",
    "South Africa": "Johannesburg",
    "New Zealand": "Auckland",
    "Morocco": "Casablanca",
    "Sri Lanka": "Colombo",
    "India": "Mumbai",
    "China": "Shanghai",
}

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
]

ITEMS_WITH_PRICES = [
    ("apple", 2), ("bread", 3), ("cheese", 5), ("milk", 4),
    ("rice", 6), ("pasta", 3), ("eggs", 4), ("butter", 5),
    ("juice", 3), ("chicken", 8), ("fish", 10), ("yogurt", 2),
    ("cereal", 4), ("coffee", 7), ("tea", 3), ("sugar", 2),
]

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]

HOBBIES = [
    "reading", "swimming", "painting", "cooking", "cycling",
    "running", "chess", "gardening", "photography", "writing",
]


def _sample(pool, n=1):
    return random.sample(pool, min(n, len(pool)))


# ---------------------------------------------------------------------------
# Task 1: Fact Correction
# ---------------------------------------------------------------------------

def generate_fact_correction() -> dict[str, Any]:
    country = random.choice(list(COUNTRIES_CAPITALS.keys()))
    correct = COUNTRIES_CAPITALS[country]
    wrong = WRONG_CAPITALS[country]

    prompt = f"The capital of {country} is"
    output_prefix = f" {wrong}. {wrong} is a major city in {country} known for"
    update = f"Correction: the capital is actually {correct}, not {wrong}."
    revised_continuation = (
        f" {correct}. {correct} is the capital city of {country}."
    )

    return {
        "prompt": prompt,
        "output_prefix": output_prefix,
        "update": update,
        "revised_continuation": revised_continuation,
        "task_type": "fact_correction",
        "expected_values": {
            "correct_value": correct,
            "wrong_value": wrong,
        },
    }


# ---------------------------------------------------------------------------
# Task 2: Variable Substitution
# ---------------------------------------------------------------------------

def generate_variable_substitution() -> dict[str, Any]:
    var_name = random.choice(["x", "y", "n", "k"])
    old_val = random.randint(2, 15)
    new_val = random.randint(2, 15)
    while new_val == old_val:
        new_val = random.randint(2, 15)

    op = random.choice([
        (f"3*{var_name} + 2", lambda v: 3 * v + 2),
        (f"2*{var_name} - 1", lambda v: 2 * v - 1),
        (f"{var_name}*{var_name}", lambda v: v * v),
        (f"5*{var_name} + 10", lambda v: 5 * v + 10),
        (f"4*{var_name} - 3", lambda v: 4 * v - 3),
    ])
    expr_str, expr_fn = op

    old_result = expr_fn(old_val)
    new_result = expr_fn(new_val)

    prompt = f"Given {var_name} = {old_val}, compute {expr_str} step by step."
    output_prefix = (
        f" Step 1: {var_name} = {old_val}."
        f" Step 2: substituting, we get"
    )
    update = f"Correction: {var_name} = {new_val}, not {old_val}."
    revised_continuation = (
        f" Step 1: {var_name} = {new_val}."
        f" Step 2: substituting into {expr_str}, we get {new_result}."
        f" The answer is {new_result}."
    )

    return {
        "prompt": prompt,
        "output_prefix": output_prefix,
        "update": update,
        "revised_continuation": revised_continuation,
        "task_type": "variable_substitution",
        "expected_values": {
            "correct_value": str(new_result),
            "wrong_value": str(old_result),
            "variable": var_name,
            "new_assignment": str(new_val),
        },
    }


# ---------------------------------------------------------------------------
# Task 3: Constraint Revision
# ---------------------------------------------------------------------------

def generate_constraint_revision() -> dict[str, Any]:
    old_budget = random.choice([50, 60, 80, 100, 120, 150])
    new_budget = random.choice([b for b in [30, 40, 50, 60, 70] if b < old_budget])

    all_items = random.sample(ITEMS_WITH_PRICES, 6)

    def pick_items(budget):
        chosen = []
        total = 0
        for name, price in all_items:
            if total + price <= budget:
                chosen.append((name, price))
                total += price
        return chosen, total

    old_items, old_total = pick_items(old_budget)
    new_items, new_total = pick_items(new_budget)

    old_list = ", ".join(f"{n} (${p})" for n, p in old_items)
    new_list = ", ".join(f"{n} (${p})" for n, p in new_items)

    prompt = f"Plan a shopping list with a budget of ${old_budget}."
    output_prefix = f" With ${old_budget}, I can buy: {old_list}. Total: ${old_total}."
    update = f"Budget changed to ${new_budget}."
    revised_continuation = (
        f" With the revised budget of ${new_budget}, I can buy: {new_list}."
        f" Total: ${new_total}."
    )

    return {
        "prompt": prompt,
        "output_prefix": output_prefix,
        "update": update,
        "revised_continuation": revised_continuation,
        "task_type": "constraint_revision",
        "expected_values": {
            "correct_value": str(new_budget),
            "wrong_value": str(old_budget),
        },
    }


# ---------------------------------------------------------------------------
# Task 4: Arithmetic Correction
# ---------------------------------------------------------------------------

def generate_arithmetic_correction() -> dict[str, Any]:
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    a_new = random.randint(10, 99)
    while a_new == a:
        a_new = random.randint(10, 99)

    op_name, op_sym, op_fn = random.choice([
        ("sum", "+", lambda x, y: x + y),
        ("difference", "-", lambda x, y: x - y),
        ("product", "*", lambda x, y: x * y),
    ])

    old_result = op_fn(a, b)
    new_result = op_fn(a_new, b)

    prompt = f"Compute the {op_name} of {a} and {b}."
    output_prefix = f" {a} {op_sym} {b} = {old_result}. The {op_name} is {old_result}."
    update = f"Correction: the first number is {a_new}, not {a}."
    revised_continuation = (
        f" {a_new} {op_sym} {b} = {new_result}. The {op_name} is {new_result}."
    )

    return {
        "prompt": prompt,
        "output_prefix": output_prefix,
        "update": update,
        "revised_continuation": revised_continuation,
        "task_type": "arithmetic_correction",
        "expected_values": {
            "correct_value": str(new_result),
            "wrong_value": str(old_result),
        },
    }


# ---------------------------------------------------------------------------
# Task 5: Key-Value Update
# ---------------------------------------------------------------------------

def generate_key_value_update() -> dict[str, Any]:
    name = random.choice(NAMES)
    old_age = random.randint(18, 65)
    new_age = random.randint(18, 65)
    while new_age == old_age:
        new_age = random.randint(18, 65)

    color = random.choice(COLORS)
    hobby = random.choice(HOBBIES)
    new_hobby = random.choice([h for h in HOBBIES if h != hobby])

    prompt = (
        f"Generate a profile: name={name}, age={old_age},"
        f" favorite_color={color}, hobby={hobby}."
    )
    output_prefix = (
        f" name: {name}, age: {old_age},"
        f" favorite_color: {color}, hobby: {hobby}."
    )
    update = f"Update: age={new_age} and hobby={new_hobby}."
    revised_continuation = (
        f" name: {name}, age: {new_age},"
        f" favorite_color: {color}, hobby: {new_hobby}."
    )

    return {
        "prompt": prompt,
        "output_prefix": output_prefix,
        "update": update,
        "revised_continuation": revised_continuation,
        "task_type": "key_value_update",
        "expected_values": {
            "correct_value": f"age={new_age}",
            "wrong_value": f"age={old_age}",
            "extra_correct": f"hobby={new_hobby}",
        },
    }


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "fact_correction": generate_fact_correction,
    "variable_substitution": generate_variable_substitution,
    "constraint_revision": generate_constraint_revision,
    "arithmetic_correction": generate_arithmetic_correction,
    "key_value_update": generate_key_value_update,
}


def generate_sample(
    task_types: list[str] | None = None,
) -> dict[str, Any]:
    """Generate one random sample from the available task types."""
    types = task_types or list(TASK_GENERATORS.keys())
    task_type = random.choice(types)
    return TASK_GENERATORS[task_type]()


def generate_dataset(
    n_samples: int,
    task_types: list[str] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate a full dataset of n_samples with balanced task types."""
    if seed is not None:
        random.seed(seed)
    types = task_types or list(TASK_GENERATORS.keys())
    samples = []
    for i in range(n_samples):
        task_type = types[i % len(types)]
        samples.append(TASK_GENERATORS[task_type]())
    random.shuffle(samples)
    return samples
