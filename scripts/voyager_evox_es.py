from dfbench import (
    EvoxES,
    ConstrainedVoyagerProblem,
    create_parser,
)


def main():
    """Run Voyager optimization using EvoX ES algorithm."""
    params = {
        "batch_size": 125,
        "pop_size": 500,
        "wall_times": [60, 120],
    }
    parser = create_parser(params, description="Voyager ES Optimization")
    args = vars(parser.parse_args())


    vp = ConstrainedVoyagerProblem()

    optimizer = EvoxES(problem=vp, batch_size=args.pop("batch_size"), variant="SNES")

    optimizer.optimize(
        save_to_file=True, 
        **args,
    )


if __name__ == "__main__":
    main()
