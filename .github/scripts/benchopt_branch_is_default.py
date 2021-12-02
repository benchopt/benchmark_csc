from pathlib import Path


MAIN_YML = ".github/workflows/main.yml"

main_yml = Path(MAIN_YML)

benchopt_branch = [
    (idx, line.split(":", 1)[1].strip())
    for idx, line in enumerate(main_yml.read_text().splitlines())
    if "BENCHOPT_BRANCH:" in line
]
print(f"benchopt_branch = '{benchopt_branch}'")

assert len(benchopt_branch) == 1
line, benchopt_branch = benchopt_branch[0]

# Issue an error if the branch used is not the master branch from benchopt
if benchopt_branch != "benchopt:main":
    print(
        "::error file={MAIN_YML},line={line}::Not default benchopt branch."
    )
    raise SystemExit(1)
