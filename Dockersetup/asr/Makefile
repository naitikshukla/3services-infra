app_name = distil_whisper_gunicorn

build:
    @docker build -t $(app_name) .

run:
    docker run --detach -p 8003:8003 $(app_name)

kill:
    @echo 'Killing container...'
    @docker ps | grep $(app_name) | awk '{print $$1}' | xargs docker


# # build Docker image
# make build
# # run the container
# make run
# # destroy it
# make kill