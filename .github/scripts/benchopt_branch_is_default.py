from pathlib import Path


MAIN_YML = ".github/workflows/main.yml"

main_yml = Path(MAIN_YML)

benchopt_branch = [
    (idx, line.split(":", 1)[1].strip())
    for idx, line in enumerate(main_yml.read_text().splitlines())
    if "benchopt_branch:" in line
]

for line, benchopt_branch in benchopt_branch:
    print(f"Using benchopt_branch = '{benchopt_branch}'")

    # Issue an error if the branch used is not the master branch from benchopt
    if benchopt_branch != "benchopt@main":
        print(
            f"::error file={MAIN_YML},line={line}::"
            "Commit using non-default branch of benchopt."
        )
        raise SystemExit(1)
