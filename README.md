REU 2025 Oyster Detection Model Training and Website deployment
Includes: Training data, Training Files, Model Comparison, Model Exports, and Server Deployment.

Website uses node.js yarn.
"yarn install" to install dependencies
"yarn start" to start locally hosted website
"yarn start --host" to start hosted website over network

Docker file is included for google cloud run server deployment

Weaknesses:
Server is set up over port 80, so web camera only works with locally hosted sever because browser security blocks and request to open camera over an unsecured port 80(http).
Google automaticlly handles https(8080) but since the server starts on port 80 it breaks.

Video is not smooth because the program does model inference every 10 frames so it causes a slideshow jittery effect.
