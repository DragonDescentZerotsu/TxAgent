"""Prediction profiles and templates shared by flat inference and query priors."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml

ROOT = Path(__file__).parent
TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction", "ames", "dili", "carcinogens")
ARCHIVE = ROOT.parents[1] / "archive/branches/prompts"


@lru_cache(maxsize=None)
def task_profiles(task):
    if task not in TASKS:
        raise ValueError(f"Unknown prediction task: {task!r}")
    document = yaml.safe_load((ROOT / "tasks" / task / "profiles.yaml").read_text())
    if document.get("task") != task or not document.get("flat"):
        raise ValueError(f"Invalid prediction profiles for {task}")
    return document


def get_profile(task, name=None):
    document = task_profiles(task)
    name = name or document["default"]
    try:
        fields = deepcopy(document["profiles"][name])
    except KeyError as error:
        raise ValueError(f"Unknown {task} prompt profile: {name!r}") from error
    for key, value in fields.items():
        if key.endswith(("_instructions", "_required_fields")):
            fields[key] = tuple(value)
        elif key.endswith("_allowed_values"):
            fields[key] = {field: set(values) for field, values in value.items()}
    return SimpleNamespace(**fields)


def bundle_directory(name):
    if Path(name).name != name:
        raise ValueError(f"Invalid prediction bundle: {name!r}")
    for parent in (ROOT / "flat", ROOT / "variants", ARCHIVE):
        directory = parent / name
        if directory.is_dir():
            return directory
    raise ValueError(f"Missing prediction bundle: {name!r}")


def flat_task_profiles(name):
    directory = bundle_directory(name)
    frozen = directory / "tasks.yaml"
    if frozen.is_file():
        return yaml.safe_load(frozen.read_text())
    tasks = {task: deepcopy(task_profiles(task)["flat"]) for task in TASKS}
    overrides = directory / "task_overrides.yaml"
    if overrides.is_file():
        for task, fields in yaml.safe_load(overrides.read_text()).items():
            tasks[task].update(fields)
    return tasks


def template_environment():
    environment = Environment(
        loader=FileSystemLoader([str(ROOT / "flat"), str(ROOT / "variants"), str(ARCHIVE), str(ROOT)]),
        undefined=StrictUndefined, autoescape=False,
    )
    environment.filters["as_json"] = lambda value: json.dumps(value, ensure_ascii=False)
    return environment


def native_prior_messages(task, query, chembl_context=None, *, prompt_profile=None):
    profile = get_profile(task, prompt_profile)
    instructions = ["Call molecule_properties for the query molecule before analysis.", *profile.single_instructions]
    if query.get("prefetched_molecule_properties"):
        instructions[0] = "Use the harness-prefetched molecule_properties result."
        if query.get("identity_hidden"):
            instructions[0] += " Do not identify or name the query."
    payload = {"task": profile.single_task, "query": query, "instructions": instructions,
               "required_json_schema": profile.single_schema}
    if chembl_context:
        instructions.append(profile.exact_context_instruction)
        payload["exact_query_chembl_context"] = chembl_context
    environment = template_environment()
    return [{"role": "system", "content": environment.get_template("query_prior/native_system.jinja").render(profile=profile, query=query)},
            {"role": "user", "content": environment.get_template("query_prior/user.jinja").render(payload=payload)}]


def query_prior_messages(contract, query):
    data = yaml.safe_load((ROOT / "query_prior/profile.yaml").read_text())
    schema = data["required_json_schema"]
    schema["endpoint_prior"] = f"{contract['positive_prediction']} | {contract['negative_prediction']} | mixed_or_unclear"
    payload = {"task": f"Assess the query molecule's structural and physicochemical prior for {contract['endpoint_name']}.",
               "query": query, "instructions": data["instructions"], "required_json_schema": schema}
    environment = template_environment()
    return [{"role": "system", "content": environment.get_template("query_prior/system.jinja").render(contract=contract)},
            {"role": "user", "content": environment.get_template("query_prior/user.jinja").render(payload=payload)}]
