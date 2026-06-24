# Ollama-Experiments-BeagleAI

Experiments done using a Beagle-AI64 board

To run the experiments locally you need first to setup python libraries, pls execute these stepts first:

- python -m venv venv
- source venv/bin/activate
- pip install matplotlib numpy httpx datasets

you can thus run the program with this command:

python benchmark_ollama.py   --model TinyLlama   --temps 0,0.1,0.2,0.3,0.4,0.5   --users 1,2,3,4,5,6   --requests-per-user 50

note that you can use different models by changing the parameter --model, different temperatures by changing --temps, add/decrease simulated active users by changing --users, and number of calls done by every user in each round by changing --request-per-user
