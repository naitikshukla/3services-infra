
## Run this to test individual docker service locally

### 1. Make Sure you can access nvidia-smi inside Docker container.
If not follow this [link](https://linuxhint.com/install-nvidia-drivers-debian-11/) and try to setup first.

### 2. Build the Docker image using the Dockerfile
make sure you are in `Dockersetup/llm` directory  
`docker build -t my_llm_image -f Dockerfile_llm .`

### 3. Currently make sure Folder structure looks like this:
- Dockersetup
    - llm
        - setup
            - ollama (command: `curl -L https://ollama.com/download/ollama-linux-amd64 -o ./setup/ollama \ chmod +x ./setup/ollama;` )
            - models
            - Modelfile(Custom model download card)
    - asr
    - gui

### 4. Run Docker Image to start ollama serve
Command:
`docker run --name ollama_serve --gpus all --rm -itd -p 11435:11434 -v ./setup:/app/setup my_llm_image`

### 5. Pull model into Docker container

Now we have started ollama server , next task is to pull any LLM model into our system, This will download model card and weights for specific model into Disks.

Location of model to download weight have been modified to use this location via env variable in Dockerfile: 

`OLLAMA_MODELS=/app/setup/models`

Consider our model is llama3:latest (which is 8B instruct model with 4.7GB weight)

* with default model prompt and param

    `docker exec -it ollama_serve ./setup/ollama run llama3`
* With custom ModelCard 

    `docker exec -it ollama_serve ./setup/ollama run ./setup/custom-llama3`

###-------------
# Build the Docker image using the Dockerfile
docker build -t my_llm_image -f Dockerfile_llm .

# Run a container from the built image
docker run --gpus all --rm -it my_llm_image


docker exec -it 7e18e4aa674b bash

docker build -t my_llm_image -f Dockerfile_llm .
docker run --gpus all --rm -itd -p 11435:11434 -v ./setup:/app/setup my_llm_image

curl http://get.docker.com | sh \ && sudo systemctl --now enable docker



./ollama create custom-llama3 -f /app/setup/Modelfile
./ollama run custom-llama3

### add user to docker group
sudo usermod -aG docker <username>

#### installing driver on debian 11
https://linuxhint.com/install-nvidia-drivers-debian-11/



### install docker ollama

### running ollama with model in docker
docker exec -it ollama ollama run llama3