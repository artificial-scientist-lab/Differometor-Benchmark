import cProfile
import pstats

from dfbench import (
    EvoxPSO,
    VoyagerProblem,
    create_parser,
)


def main():
    """Run Voyager optimization using EvoX PSO algorithm."""
    params = {
        "batch_size": 50,
        "pop_size": 50,
        # "n_generations": 10,
        "wall_times": [60],
    }
    parser = create_parser(params, description="Voyager PSO Optimization")
    args = vars(parser.parse_args())

    # Initialize problem
    vp = VoyagerProblem()

    # Initialize optimizer with batch_size to control memory usage
    # Start with batch_size=5, increase if you have memory to spare
    optimizer = EvoxPSO(problem=vp, batch_size=args.pop("batch_size"))

    optimizer.optimize(
        **args,
    )


if __name__ == "__main__":
    main()
