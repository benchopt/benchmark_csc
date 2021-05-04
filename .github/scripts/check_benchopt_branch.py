from pathlib import Path


main_yml = Path(".github/workflows/main.yml")

benchopt_branch = [
    (idx, line.split(":", 1)[1].strip())
    for idx, line in enumerate(main_yml.read_text().splitlines())
    if "BENCHOPT_BRANCH:" in line
]
print(f"benchopt_branch = '{benchopt_branch}'")

assert len(benchopt_branch) == 1
line, benchopt_branch = benchopt_branch[0]

# Issue a warning if the branch used is not the master branch from benchopt
if benchopt_branch != "benchopt:master":
    print(
        "::warning file={main_yml},line={line}::Not default benchopt branch."
    )
