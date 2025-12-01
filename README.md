This is a repository for NYU HPC chatbot.
(latest version for tests; NYU-HPC-Assistant1 supports a chatbot outside NYU; NYU-HPC-Assistant3 supports a chatbot inside NYU)


# create a conda environment

`conda create -n chatbot-py310 python==3.10`

`conda activate chatbot-py310`

`pip install -r requirements.txt`

or better for the specific versions,

`pip install -r requirements-specific.txt`

# set up the environment for API KEYs

The API KEYs for foundation language models must be exported first. These keys are needed in both preparing and starting the chatbot.

To import the API KEYs,

`export JINA_API_KEY=xxx`

`export OPENAI_API_KEY=xxx` (if use portkey API the format is different)


# generate data files for the chatbot

This step can be skipped if the model is constructed. Otherwise, use the following command to generate data

`python main.py`

Since this step takes, better put it in a script and submit it as a slurm job.

# start the chatbot

After it, then start the chatbot with this command,

`python -m streamlit run --server.port 8080 streamlit_app.py`

or the cloud version where files are stored in cloud (e.g., AWS, GCP, etc),

`python -m streamlit run --server.port 8080 streamlit_app.py`

## some notes
Packages is critical
(my NYU MacBook)
conda create -n chatbot2 python==3.10
need numpy<=2.0:
pip install "numpy<2"

different versions of Python, FAISS, etc. serious affect the functionality of the code. How to control the verion of code? Prepare a requirement file.

tf-keras

Faiss related issues, a popular one due to using different faiss.swigfaiss_avx512 or faiss.swigfaiss (TypeError: IndexFlat.search() missing 3 required positional arguments: 'k', 'distances', and 'labels’)
Solution is to uninstall the old packages and use the standard one:

pip uninstall faiss
pip uninstall faiss-cpu
pip uninstall faiss-gpu
conda remove faiss faiss-cpu faiss-gpu
conda install -c conda-forge faiss-cpu

if another issue appear: could not find faiss.swigfaiss_avx2 or faiss.swigfaiss_avx512
in file 
import sys
sys.modules['faiss.swigfaiss_avx2'] = faiss
sys.modules['faiss.swigfaiss_avx512'] = faiss

JINA API

this needs to be updated when there is related issues. You can check the latest API keys here: https://jina.ai/api-dashboard/embedding

The key can be set in a terminal (the local version) or explicitly (the cloud version) in a secret session or directly in the code (not recommended for security, some platform may even forbit it).

LLM API Keys

there is personal version and NYU provided ones. Different rules apply. Need to be careful to check the format differences and VPN is needed for NYU KEYS through PortKey.

Foundation model version:

gpt-4o, got-4o-mini

AWS S3 bucket

homemade cloud version

Google cloud

Data storage and priority management
Burwood version, and Gemini

Tunneling from compute node, localhost
(i) first, start a short srun job;

(ii) then open a new terminal and try the following:
### On your local machine
ssh -L 8080:localhost:8080 your_netid@login.hpc.nyu.edu
### Then, from inside login node:
ssh -L 8080:localhost:8080 node123

(iii) After this, log in to the node 123 and start the application app.py

# upload local files to a GitHub repo

set up steps

0 ininitialize and generate a .git/ folder, which is used to connect with github

`git init -b main`

1 create an empty repository and get its name

`git@github.com:peizong/NYU-HPC-Assistantxx.git`

2 connect this empy repository with local file

`git remote add origin git@github.com:peizong/NYU-HPC-Assistantx.git`

`git push -u origin main`

3 generate a key-like authorization

`gh auth login` (need to install gh: brew install gh)

4 github push files

`git add sth`

`git commit -m "First Commit"`

`git remote add origin git@github.com:peizong/NYU-HPC-Assistantxx.git`

`git branch -M main`

`git push -u origin main`

# host the chatbot in cloud (e.g., OpenShift)

clean steps to set up Red Hat OpenShift

  0 in the folder LLM_UI, create a ".s2i/bin" folder and "run" file and put the following line in the run file:

  APP_PORT=${APP_PORT:-"8080"}
  
  streamlit run --server.port ${APP_PORT} streamlit_app_cloud.py

  1 open https://console.cloud.rt.nyu.edu/, choose “developer” view, choose “Import from Git” [from “Git Repo” NOT “Serverless function"]
  
  2 Git Repo URL: https://github.com/peizong/xxx.git
  
  3 Show advanced Git options/Context dir: LLM_UI (for the Python version, I used the default Python 3.12 (UBI 8))
  
  4 General/name: hpc-chatbot3 -> chatbot
  
  5 Show advanced Build option/Environment variables (build and runtime)
  
   JINA_API_KEY               jina_xxx
   
   PORTKEY_API_KEY      xxx
   
   VIRTUAL_KEY_VALUE  openai-nyu-it-d-xxx
   
  6  Deploy/Resource type: Deployment
  
  #7 Show advanced Routing options/Hostname: hoc-chatbot3 -> chatbot3.apps.cloud.rt.nyu.edu -> chatbot.apps.cloud.rt.nyu.edu
  
  8 Create

possible reasons for issues: 
(1) missing three parameter in a faiss function: numpy version, faiss version, force install faiss, 
(2) no output: API KEY, the different output of port key and open AI key;
(3) when building  is slow, add cpus and memory (choose Actions/Resource to request CPUs/Memory for building  or add the following line in the YAML file, resources part)
    requests:
      cpu: 2
      memory: 40Gi
