# codif-ape-graph-rag

Create a .env file that should have those:

URL_EMBEDDING_API=****
EMBEDDING_MODEL = ****
GENERATION_MODEL = ****
URL_LLM_API = ********

LANGFUSE_PUBLIC_KEY=****
LANGFUSE_SECRET_KEY=****
LANGFUSE_HOST=****
OPENAI_API_KEY=****
MLFLOW_TRACKING_URI=****


## Local run 🏛️
You need to install to first install [**uv**](https://github.com/astral-sh/uv), using for instance `pip install uv` if applicable.

Please reach out to us to get our [**Neo4J**](https://neo4j.com/) graph database's password. Then run the following commands to locally launch the API:

```bash
. setup.sh
export NEO4J_API_KEY=[NEO4J PASSWORD]
cd src
uv run uvicorn api.main:app --host 0.0.0.0 --port 5000
```

To locally launch the web application, run the following command, on another terminal:

```bash
cd src
uv run streamlit run main.py
```

## Deployment 🚀

The web application is available [here](https://codification-ape-graph-rag.lab.sspcloud.fr/), and the API swagger [there](https://codification-ape-graph-rag.lab.sspcloud.fr/api/docs).

We used:
- [**supervisord**](https://supervisord.org/) for double deployment of the API and the web application, on two different custom ports 🖇️
- **Docker** for conteneurization 🐳 (find the image repo [here](https://hub.docker.com/repository/docker/meilametayebjee/codification-ape-graph-rag-api/general))
- **GitHub Actions** for continous integration ♾️
- and **ArgoCD** for continous deployment 🚀

We used our [datalab](https://datalab.sspcloud.fr/) powered by [**Onyxia**](https://www.onyxia.sh/) and a **Kubernetes** cluster to deploy the application.


## Evaluation

In order to run evaluation of different methods you can execute the following commands:

```bash
export MLFLOW_TRACKING_URI=https://projet-ape-mlflow.user.lab.sspcloud.fr
uv run evaluate.py --num_samples 1000 --entry_point all
```
