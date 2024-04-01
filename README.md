# Photosmyth

[**Photosmyth**](https://prikshit.dev/projects/photosmyth/) is a generative AI application built using streamlit that generates images.
To build this application, I used [`sdxl-turbo`](https://build.nvidia.com/stabilityai/sdxl-turbo), a fast text-to-image model provided by [NVidia AI Foundation Models and Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/). It is hosted on [NVidia API Catalog](https://build.nvidia.com/explore/discover) which provides easy access to the model APIs.

You can read more about photosmyth and the development process [here](https://prikshit.dev/blog/photosmyth).

![](./assests/video.gif)

## Getting started

Step 1: Create a virtual environment

`python3 -m virtualenv venv`

Step 2: Activate the virtual environment

`source venv/bin/activate`

Step 3: Install the required dependencies

`pip install -r requirements.txt`

Step 4: Start the streamlit app

`streamlit run app.py --server.port 8013`

Navigate to `http://localhost:8013` on your browser.