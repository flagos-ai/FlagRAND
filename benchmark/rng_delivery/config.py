"""Fixed shapes and native-generator identities for RNG delivery."""

from benchmark.muxi_matrix.config import Profile
from benchmark.nvidia_matrix.config import NORMAL_MATRIX_SHAPES, RAW_UNIFORM_MATRIX_SHAPES

ENGINES = ("xorwow", "mrg32k3a", "mtgp32")
APIS = ("raw32", "uniform_f32", "uniform_f64", "normal_f32", "normal_f64", "lognormal_f32", "lognormal_f64")
POISSON_ENGINES = ("philox",) + ENGINES
OPERATORS = tuple(f"{engine}_{api}" for engine in ENGINES for api in APIS) + tuple(
    f"{engine}_poisson" for engine in POISSON_ENGINES
)
COREX_OPERATORS = tuple(operator for operator in OPERATORS if not operator.endswith("_f64"))
OPERATOR_GROUPS = {engine: tuple(f"{engine}_{api}" for api in APIS + ("poisson",)) for engine in ENGINES}
OPERATOR_GROUPS["philox"] = APIS + ("philox_poisson",)
OPERATOR_GROUPS["poisson"] = tuple(f"{engine}_poisson" for engine in POISSON_ENGINES)
NATIVE_TYPES = {"philox": 161, "xorwow": 101, "mrg32k3a": 121, "mtgp32": 141}
LAMBDAS = (0.1, 1.0, 10.0, 30.0, 1000.0)
ODD_TAILS = (1, 3, 127, 129, 1023, 1025, 4097)
PROFILES = {
    "smoke": Profile("smoke", (1024, 65536, 1048576), 8, 5, 80.0),
    "matrix": Profile("matrix", (), 5, 11, 200.0),
}


def split_operator(operator):
    return operator.split("_", 1)


def shapes_for(api, profile):
    if profile == "smoke":
        return PROFILES[profile].sizes
    return NORMAL_MATRIX_SHAPES if api.startswith(("normal", "lognormal")) else RAW_UNIFORM_MATRIX_SHAPES


def parameter_values(api):
    return LAMBDAS if api == "poisson" else (None,)
