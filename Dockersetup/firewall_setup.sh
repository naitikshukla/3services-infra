# sudo apt-get install ufw
# sudo ufw enable

# Allow traffic within Docker network
sudo ufw allow in on Dockerfile_gui

# Allow traffic between Docker containers
sudo ufw allow in on Dockerfile_gui from any to any

# Allow traffic to Service 3 (GUI and Logic)
sudo ufw allow in on Dockerfile_gui from any to any port 5000

# Deny traffic to Service 1 and Service 2 from outside Docker network
sudo ufw deny in on Dockerfile_gui from any to Dockerfile_llm
sudo ufw deny in on Dockerfile_gui from any to Dockerfile_asr
