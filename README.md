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
