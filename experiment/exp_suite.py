from collections import namedtuple
import itertools

# modify to something usable
BASE_MODEL = {
        "S1": "huggyllama/llama-7b",
        "S4": "dummy-llama-13b",
        "Real": "huggyllama/llama-7b",
}

LORA_DIR = {
        "S1": ["dummy-lora-7b-rank-8"],
        "S4": ["dummy-lora-13b-rank-8"],
        "Real": ["tloen/alpaca-lora-7b"],
}

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    ["num_adapters",
     "alpha", # power law distribution for lambda_i, which are the mean rate for poisson arrival process
     "req_rate", # total request rate per second
     "cv", # coefficient of variation. When cv == 1, the arrival process is Poisson process.
     "duration", # benchmark serving duration
     "input_range", # input length l.b. and u.b.
     "output_range", # output length l.b. and u.b.
     "on_off",
     "mode",
    ]
)


paper_suite = {
    "ablation_lshare": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.2, 0.8]],
        cv = [-1],
        duration = [60*5],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "real": BenchmarkConfig(
        num_adapters = [-1],
        alpha = [-1],
        req_rate = [3.5],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[1, 1024]],
        output_range = [[1, 1024]],
        on_off = [-1],
        mode = ["real"],
    ),
    "overload": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[1.5, 3]],
        cv = [-1],
        # duration = [60 * 6],
        duration = [60], #debug
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-multi": BenchmarkConfig(
        num_adapters = [8],
        alpha = [-1],
        req_rate = [[0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6]],
        cv = [-1],
        duration = [60 * 6],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-weighted": BenchmarkConfig(
        num_adapters = [4],
        alpha = [-1],
        req_rate = [[0.8, 0.8, 1.2, 1.6]],
        cv = [-1],
        duration = [60 * 6],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[4, 8]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4-short": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[8, 16]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[128, 129]],
        output_range = [[128, 129]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4-long": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[2, 4]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[512, 513]],
        output_range = [[512, 513]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4-35000-128": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[6, 12]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[128, 129]],
        output_range = [[128, 129]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4-35000-256": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[3, 6]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4-35000-512": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[1.5, 3]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[512, 513]],
        output_range = [[512, 513]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload-s4-35000-768": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[1, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[768, 769]],
        output_range = [[768, 769]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "proportional": BenchmarkConfig(
        num_adapters = [3],
        alpha = [-1],
        req_rate = [[0.25, 0.5, 1.5]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "increase": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["increase"],
    ),
    "on_off_less": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [60],
        mode = ["uniform"],
    ),
    "on_off_overload": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[2, 3]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [60],
        mode = ["uniform"],
    ),
    "poisson_on_off_overload": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[16, 512]],
        output_range = [[16, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),
    "poisson_short_long": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[8, 1.5]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["poisson-short-long"],
    ),
    "poisson_short_long_2": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[8, 1.5]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[32, 33]],
        output_range = [[512, 513]],
        on_off = [-1],
        mode = ["poisson-short-long-2"],
    ),
    "dist_shift": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[-1, -1]],
        cv = [-1],
        duration = [60 * 15],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["dist_shift"],
    ),
}


debug_suite = {
    "default": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 1],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "diff_slo": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "increase": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [-1],
        mode = ["increase"],
    ),

    "on_off": BenchmarkConfig(
        num_adapters = [2],
        alpha = [0.4],
        req_rate = [3],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "on_off_equal": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [3],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "unbalance": BenchmarkConfig(
        num_adapters = [2],
        alpha = [0.2],
        req_rate = [4],
        cv = [1],
        duration = [60 * 20],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [-1],
        mode = ["poisson"],
    ),
}


def get_all_suites(debug=False, suite=None, breakdown=False):
    assert not (debug and breakdown)
    assert suite is not None
    if debug:
        exps = [{suite: debug_suite[suite]}]
    # elif breakdown:
    #     exps = [{suite: breakdown_suite[suite]}]
    else:
        exps = [{suite: paper_suite[suite]}]

    suites = []
    for exp in exps:
        for workload in exp:
            (num_adapters, alpha, req_rate, cv, duration,
                    input_range, output_range, on_off, mode) = exp[workload]
            if mode == "real":
                # These arguments are not used in real trace
                num_adapters = alpha = cv = [None]

            for combination in itertools.product(
                                   num_adapters, alpha, req_rate, cv, duration,
                                   input_range, output_range, on_off, mode):
                suites.append(combination)
    return suites


def to_dict(config):
    ret = {}
    for i, key in enumerate(BenchmarkConfig._fields):
        ret[key] = config[i]
    return ret