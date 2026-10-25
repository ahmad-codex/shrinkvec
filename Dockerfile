# The CLI without installing Python:
#
#   docker run --rm --network host -e SHRINKVEC_DSN ghcr.io/ahmad-codex/shrinkvec \
#     run --table public.documents --column embedding
FROM python:3.12-slim
COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir "/src[pg,datasets]" && rm -rf /src
WORKDIR /work
ENTRYPOINT ["shrinkvec"]
CMD ["--help"]
