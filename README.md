### Docker and Apptainer setup
Build using Docker: `docker build -t <name> .`

Run if the container using Docker: `docker run -it --rm -v ./results:/app/results <name> bash`. Only runs a shell as an example.

Save docker image to a `.tar` file: `docker save <name> -o <other-name>.tar`

Get an `apptainer` image from the original docker image: `apptainer build <other-name>.sif docker-archive://<name>.tar`


Run the container using `apptainer`:
```
apptainer exec --nv --containall \
    --bind ./results:/app/results \
    testname.sif \
    bash -c "cd /app/src/main && uv run --env-file .env --no-sync main.py"
```

### Run the project

Inside `src/main`:

```uv run --env-file .env main.py```



### Setup for configurations:

Before:

```
train/
    <id>/
        eval_results/
        logs/
            (training info)
        model.zip
        training_metadata.json

metafeatures/
    ENV_<obs-id>_OBS_<obs-id>
```

New setup:
```
<ENV_NAME>/
    <random-id>/
        instance_config.json # Important: information about config: environment and observation configurations - does not include algorithm information
        train/
            <algo-random-id>/
                algo_config.json # config about algorithm
                logs/
                    (training info)
                model.zip
                training_metadata.json
                eval_results.json
        metafeatures/
            ...
```

From config information to id: $O(N)$
From id to config information: $O(1)$

Requires an $O(N)$ check for each configuration to check if it has been trained or not.
This can be costly, if we need to filter them out all initially (even when not training all of them).
Therefore, it makes sense to filter them out only just before starting training instead of always at the start.

If I use a deterministic algorithm, config information to id would work in $O(1)$ but would break if I change anything, so this O(N) solution might be more flexible.
