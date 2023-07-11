"""Functions for registering environments within gymnasium using public functions ``make``, ``register`` and ``spec``."""
from __future__ import annotations

import contextlib
import difflib
import importlib
import importlib.metadata as metadata   
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, Protocol

from complete_playground import Env, error, logger

ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)

__all__ = [
    "EnvSpec",
    "registry",
    "current_namespace",
    "register",
    "make",
    "spec",
    "pprint_registry",
]

class EnvCreator(Protocol):
    """Function type expected for an environment."""

    def __call__(self, **kwargs: Any) -> Env:
        ...

@dataclass
class EnvSpec:
    """
    A specification for creating environments with :meth:'complete_playground.make'

    * **id**: The string used to create the environment with :meth:`complete_playground.make`
    * **entry_point**: A string for the environment location, ``(import path):(environment name)`` or a function that creates the environment.
    * **reward_threshold**: The reward threshold for completing the environment.
    * **max_episode_steps**: The max number of steps that the environment can take before truncation
    """

    id: str
    entry_point: EnvCreator | str | None = field(default=None)

    # Environment attributes
    reward_threshold: float | None = field(default=None)
    max_episode_steps: int | None = field(default=None)

    # post-init attributes
    namespace: str | None = field(init=False)
    name: str = field(init=False)
    version: int | None = field(init=False)

    
    def __post_init__(self):
        """Calls after the spec is created to extract the namespace, name and version from the id."""
        # Initialize namespace, name, version
        self.namespace, self.name, self.version = parse_env_id(self.id)

    def make(self, **kwargs: Any) -> Env:
        """Calls ``make`` using the environment spec and any keyword arguments."""
        # For compatibility purposes
        return make(self, **kwargs)


# Global registry of environments. Meant to be accessed through `register` and `make`
registry: dict[str, EnvSpec] = {}
current_namespace: str | None = None


def parse_env_id(env_id: str) -> tuple[str | None, str, int | None]:
    """Parse environment ID string format - ``[namespace/](env-name)[-v(version)]`` where the namespace and version are optional.

    Args:
        env_id: The environment id to parse

    Returns:
        A tuple of environment namespace, environment name and version number

    Raises:
        Error: If the environment id is not valid environment regex
    """
    match = ENV_ID_RE.fullmatch(env_id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {env_id}. (Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
        )
    ns, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return ns, name, version


def get_env_id(ns: str | None, name: str, version: int | None) -> str:
    """Get the full env ID given a name and (optional) version and namespace. Inverse of :meth:`parse_env_id`.

    Args:
        ns: The environment namespace
        name: The environment name
        version: The environment version

    Returns:
        The environment id
    """
    full_name = name
    if ns is not None:
        full_name = f"{ns}/{name}"
    if version is not None:
        full_name = f"{full_name}-v{version}"

    return full_name


def find_highest_version(ns: str | None, name: str) -> int | None:
    """Finds the highest registered version of the environment given the namespace and name in the registry.

    Args:
        ns: The environment namespace
        name: The environment name (id)

    Returns:
        The highest version of an environment with matching namespace and name, otherwise ``None`` is returned.
    """
    version: list[int] = [
        env_spec.version
        for env_spec in registry.values()
        if env_spec.namespace == ns
        and env_spec.name == name
        and env_spec.version is not None
    ]
    return max(version, default=None)


def _check_namespace_exists(ns: str | None):
    """Check if a namespace exists. If it doesn't, print a helpful error message."""
    # If the namespace is none, then the namespace does exist
    if ns is None:
        return

    # Check if the namespace exists in one of the registry's specs
    namespaces: set[str] = {
        env_spec.namespace
        for env_spec in registry.values()
        if env_spec.namespace is not None
    }
    if ns in namespaces:
        return

    # Otherwise, the namespace doesn't exist and raise a helpful message
    suggestion = (
        difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    )
    if suggestion:
        suggestion_msg = f"Did you mean: `{suggestion[0]}`?"
    else:
        suggestion_msg = f"Have you installed the proper package for {ns}?"

    raise error.NamespaceNotFound(f"Namespace {ns} not found. {suggestion_msg}")


def _check_name_exists(ns: str | None, name: str):
    """Check if an env exists in a namespace. If it doesn't, print a helpful error message."""
    # First check if the namespace exists
    _check_namespace_exists(ns)

    # Then check if the name exists
    names: set[str] = {
        env_spec.name for env_spec in registry.values() if env_spec.namespace == ns
    }
    if name in names:
        return

    # Otherwise, raise a helpful error to the user
    suggestion = difflib.get_close_matches(name, names, n=1)
    namespace_msg = f" in namespace {ns}" if ns else ""
    suggestion_msg = f" Did you mean: `{suggestion[0]}`?" if suggestion else ""

    raise error.NameNotFound(
        f"Environment `{name}` doesn't exist{namespace_msg}.{suggestion_msg}"
    )

def _check_version_exists(ns: str | None, name: str, version: int | None):
    """Check if an env version exists in a namespace. If it doesn't, print a helpful error message.

    This is a complete test whether an environment identifier is valid, and will provide the best available hints.

    Args:
        ns: The environment namespace
        name: The environment space
        version: The environment version

    Raises:
        DeprecatedEnv: The environment doesn't exist but a default version does
        VersionNotFound: The ``version`` used doesn't exist
        DeprecatedEnv: Environment version is deprecated
    """
    if get_env_id(ns, name, version) in registry:
        return

    _check_name_exists(ns, name)
    if version is None:
        return

    message = f"Environment version `v{version}` for environment `{get_env_id(ns, name, None)}` doesn't exist."

    env_specs = [
        env_spec
        for env_spec in registry.values()
        if env_spec.namespace == ns and env_spec.name == name
    ]
    env_specs = sorted(env_specs, key=lambda env_spec: int(env_spec.version or -1))

    default_spec = [env_spec for env_spec in env_specs if env_spec.version is None]

    if default_spec:
        message += f" It provides the default version `{default_spec[0].id}`."
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)

    # Process possible versioned environments

    versioned_specs = [
        env_spec for env_spec in env_specs if env_spec.version is not None
    ]

    latest_spec = max(versioned_specs, key=lambda env_spec: env_spec.version, default=None)  # type: ignore
    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{env_spec.version}`" for env_spec in env_specs)
        message += f" It provides versioned environments: [ {version_list_msg} ]."

        raise error.VersionNotFound(message)

    if latest_spec is not None and version < latest_spec.version:
        raise error.DeprecatedEnv(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. "
            f"Please use `{latest_spec.id}` instead."
        )
    



def _check_spec_register(testing_spec: EnvSpec):
    """Checks whether the spec is valid to be registered. Helper function for `register`."""
    latest_versioned_spec = max(
        (
            env_spec
            for env_spec in registry.values()
            if env_spec.namespace == testing_spec.namespace
            and env_spec.name == testing_spec.name
            and env_spec.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            env_spec
            for env_spec in registry.values()
            if env_spec.namespace == testing_spec.namespace
            and env_spec.name == testing_spec.name
            and env_spec.version is None
        ),
        None,
    )

    if unversioned_spec is not None and testing_spec.version is not None:
        raise error.RegistrationError(
            "Can't register the versioned environment "
            f"`{testing_spec.id}` when the unversioned environment "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and testing_spec.version is None:
        raise error.RegistrationError(
            f"Can't register the unversioned environment `{testing_spec.id}` when the versioned environment "
            f"`{latest_versioned_spec.id}` of the same name already exists. Note: the default behavior is "
            "that `complete_playground.make` with the unversioned environment will return the latest versioned environment"
        )
    

def _check_metadata(testing_metadata: dict[str, Any]):
    """Check the metadata of an environment."""
    if not isinstance(testing_metadata, dict):
        raise error.InvalidMetadata(
            f"Expect the environment metadata to be dict, actual type: {type(metadata)}"
        )

    render_modes = testing_metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            f"The environment creator metadata doesn't include `render_modes`, contains: {list(testing_metadata.keys())}"
        )
    elif not isinstance(render_modes, Iterable):
        logger.warn(
            f"Expects the environment metadata render_modes to be a Iterable, actual type: {type(render_modes)}"
        )


def _find_spec(id: str) -> EnvSpec:
    module, env_name = (None, id) if ":" not in id else id.split(":")
    if module is not None:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}. Environment registration via importing a module failed. "
                f"Check whether '{module}' contains env registration and can be imported."
            ) from e

    # load the env spec from the registry
    env_spec = registry.get(env_name)

    # update env spec is not version provided, raise warning if out of date
    ns, name, version = parse_env_id(env_name)

    latest_version = find_highest_version(ns, name)
    if version is not None and latest_version is not None and latest_version > version:
        logger.warn(
            f"The environment {env_name} is out of date. You should consider "
            f"upgrading to version `v{latest_version}`."
        )
    if version is None and latest_version is not None:
        version = latest_version
        new_env_id = get_env_id(ns, name, version)
        env_spec = registry.get(new_env_id)
        logger.warn(
            f"Using the latest versioned environment `{new_env_id}` "
            f"instead of the unversioned environment `{env_name}`."
        )

    if env_spec is None:
        _check_version_exists(ns, name, version)
        raise error.Error(f"No registered env with id: {env_name}")

    return env_spec


def load_env(name: str) -> EnvCreator:
    """Loads an environment with name of style ``"(import path):(environment name)"`` and returns the environment creation function, normally the environment class type.

    Args:
        name: The environment name

    Returns:
        The environment constructor for the given environment name.
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn



def load_plugin_envs(entry_point: str = "gymnasium.envs"):
    """Load modules (plugins) using the gymnasium entry points in order to register external module's environments on ``import gymnasium``.

    Args:
        entry_point: The string for the entry point.
    """
    # Load third-party environments
    for plugin in metadata.entry_points(group=entry_point):
        # Python 3.8 doesn't support plugin.module, plugin.attr
        # So we'll have to try and parse this ourselves
        module, attr = None, None
        try:
            module, attr = plugin.module, plugin.attr  # type: ignore  ## error: Cannot access member "attr" for type "EntryPoint"
        except AttributeError:
            if ":" in plugin.value:
                module, attr = plugin.value.split(":", maxsplit=1)
            else:
                module, attr = plugin.value, None
        except Exception as e:
            logger.warn(
                f"While trying to load plugin `{plugin}` from {entry_point}, an exception occurred: {e}"
            )
            module, attr = None, None
        finally:
            if attr is None:
                raise error.Error(
                    f"Gymnasium environment plugin `{module}` must specify a function to execute, not a root module"
                )

        context = namespace(plugin.name)
        if plugin.name.startswith("__") and plugin.name.endswith("__"):
            # `__internal__` is an artifact of the plugin system when
            # the root namespace had an allow-list. The allow-list is now
            # removed and plugins can register environments in the root
            # namespace with the `__root__` magic key.
            if plugin.name == "__root__" or plugin.name == "__internal__":
                context = contextlib.nullcontext()
            else:
                logger.warn(
                    f"The environment namespace magic key `{plugin.name}` is unsupported. "
                    "To register an environment at the root namespace you should specify the `__root__` namespace."
                )

        with context:
            fn = plugin.load()
            try:
                fn()
            except Exception:
                logger.warn(f"plugin: {plugin.value} raised {traceback.format_exc()}")


@contextlib.contextmanager
def namespace(ns: str):
    """Context manager for modifying the current namespace."""
    global current_namespace
    old_namespace = current_namespace
    current_namespace = ns
    yield
    current_namespace = old_namespace



def register(
    id: str,
    entry_point: EnvCreator | str | None = None,
    reward_threshold: float | None = None,
    max_episode_steps: int | None = None,
):
    """Registers an environment in gymnasium with an ``id`` to use with :meth:`complete_playground.make` with the ``entry_point`` being a string or callable for creating the environment.

    The ``id`` parameter corresponds to the name of the environment, with the syntax as follows:
    ``[namespace/](env_name)[-v(version)]`` where ``namespace`` and ``-v(version)`` is optional.


    Args:
    id: The environment id
    entry_point: The entry point for creating the environment
    reward_threshold: The reward threshold considered for an agent to have learnt the environment
    max_episode_steps: The maximum number of episodes steps before truncation. Used by the :class:`gymnasium.wrappers.TimeLimit` wrapper if not ``None``.
    
    """
    assert (
        entry_point is not None
    ), "`entry_point` must be provided"
    global registry, current_namespace
    ns, name, version = parse_env_id(id)

    # does not have the kwargs check gymnasium/envs/registration.py has
    if current_namespace is not None:
        ns_id = current_namespace
    else:
        ns_id = ns

    full_env_id = get_env_id(ns_id, name, version)

    new_spec = EnvSpec(
        id=full_env_id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
        max_episode_steps=max_episode_steps
    )
    _check_spec_register(new_spec)

    if new_spec.id in registry:
        logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def make(
    id: str | EnvSpec,
    max_episode_steps: int | None = None,
) -> Env:
    """Creates an environment previously registered with :meth:`complete_playground.register` or a :class:`EnvSpec`.

    To find all available environments use ``complete_playground.envs.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id or a :class:`EnvSpec`. Optionally if using a string, a module to import can be included, e.g. ``'module:Env-v0'``.
            This is equivalent to importing the module first to register the environment followed by making the environment.
        max_episode_steps: Maximum length of an episode, can override the registered :class:`EnvSpec` ``max_episode_steps``.
            The value is used by :class:`gymnasium.wrappers.TimeLimit`.
        
    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist in the :attr:`registry`
    """
    if isinstance(id, EnvSpec):
        env_spec = id
    else:
        env_spec = _find_spec(id)

    assert isinstance(
        env_spec, EnvSpec
    ), f"We expected to collect an `EnvSpec`, actually collected a {type(env_spec)}"

    
    # Load the environment creator
    if env_spec.entry_point is None:
        raise error.Error(f"{env_spec.id} registered but entry_point is not specified")
    elif callable(env_spec.entry_point):
        env_creator = env_spec.entry_point
    else:
        # Assume it's a string
        env_creator = load_env(env_spec.entry_point)


    ####################
    # No rendering options yet for our environments
    ####################

    ####################
    # No api comptability application
    ####################

    env = env_creator()     # may need to come back to this to allow for kwargs

    ####################
    # No wrappers
    ####################


    return env

def spec(env_id: str) -> EnvSpec:
    """Retrieve the :class:`EnvSpec` for the environment id from the :attr:`registry`.

    Args:
        env_id: The environment id with the expected format of ``[(namespace)/]id[-v(version)]``

    Returns:
        The environment spec if it exists

    Raises:
        Error: If the environment id doesn't exist
    """
    env_spec = registry.get(env_id)
    if env_spec is None:
        ns, name, version = parse_env_id(env_id)
        _check_version_exists(ns, name, version)
        raise error.Error(f"No registered env with id: {env_id}")
    else:
        assert isinstance(
            env_spec, EnvSpec
        ), f"Expected the registry for {env_id} to be an `EnvSpec`, actual type is {type(env_spec)}"
        return env_spec
    
def pprint_registry(
    print_registry: dict[str, EnvSpec] = registry,
    *,
    num_cols: int = 3,
    exclude_namespaces: list[str] | None = None,
    disable_print: bool = False,
) -> str | None:
    """Pretty prints all environments in the :attr:`registry`.

    Note:
        All arguments are keyword only

    Args:
        print_registry: Environment registry to be printed. By default, :attr:`registry`
        num_cols: Number of columns to arrange environments in, for display.
        exclude_namespaces: A list of namespaces to be excluded from printing. Helpful if only ALE environments are wanted.
        disable_print: Whether to return a string of all the namespaces and environment IDs
            or to print the string to console.
    """
    # Defaultdict to store environment ids according to namespace.
    namespace_envs: dict[str, list[str]] = defaultdict(lambda: [])
    max_justify = float("-inf")

    # Find the namespace associated with each environment spec
    for env_spec in print_registry.values():
        ns = env_spec.namespace

        if ns is None and isinstance(env_spec.entry_point, str):
            # Use regex to obtain namespace from entrypoints.
            env_entry_point = re.sub(r":\w+", "", env_spec.entry_point)
            split_entry_point = env_entry_point.split(".")

            if len(split_entry_point) >= 3:
                # If namespace is of the format:
                #  - gymnasium.envs.mujoco.ant_v4:AntEnv
                #  - gymnasium.envs.mujoco:HumanoidEnv
                ns = split_entry_point[2]
            elif len(split_entry_point) > 1:
                # If namespace is of the format - shimmy.atari_env
                ns = split_entry_point[1]
            else:
                # If namespace cannot be found, default to env name
                ns = env_spec.name

        namespace_envs[ns].append(env_spec.id)
        max_justify = max(max_justify, len(env_spec.name))

    # Iterate through each namespace and print environment alphabetically
    output: list[str] = []
    for ns, env_ids in namespace_envs.items():
        # Ignore namespaces to exclude.
        if exclude_namespaces is not None and ns in exclude_namespaces:
            continue

        # Print the namespace
        namespace_output = f"{'=' * 5} {ns} {'=' * 5}\n"

        # Reference: https://stackoverflow.com/a/33464001
        for count, env_id in enumerate(sorted(env_ids), 1):
            # Print column with justification.
            namespace_output += env_id.ljust(max_justify) + " "

            # Once all rows printed, switch to new column.
            if count % num_cols == 0:
                namespace_output = namespace_output.rstrip(" ")

                if count != len(env_ids):
                    namespace_output += "\n"

        output.append(namespace_output.rstrip(" "))

    if disable_print:
        return "\n".join(output)
    else:
        print("\n".join(output))
