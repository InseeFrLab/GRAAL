# Déploiement de l'app de revue MatchVerifier

Manifests destinés à [`InseeFrLab/codif-ape-cd`](https://github.com/InseeFrLab/codif-ape-cd),
le dépôt GitOps lu par Argo CD. Ils déploient
[`src/evaluation/apps/match_verifier_eval_app.py`](../../src/evaluation/apps/match_verifier_eval_app.py)
dans le namespace `projet-ape`, pour que les trois annotateurs travaillent sur une
instance partagée dont les jugements atterrissent directement sur S3.

## Où va quoi

| Ici | Dans `codif-ape-cd` |
| --- | --- |
| `graal-match-verifier-review/` (deployment, service, ingress) | `graal-match-verifier-review/`, à la racine |
| `argocd-templates/graal-match-verifier-review.yaml` | `argocd-templates/` |

L'`Application` Argo CD pointe sur le dossier `graal-match-verifier-review` de
`codif-ape-cd` avec `selfHeal: true`, comme les autres apps du dépôt.

## Avant le premier sync

1. **L'image doit exister.** Elle est construite et poussée par
   `.github/workflows/docker.yml` du dépôt GRAAL à chaque push sur `main`, vers
   `inseefrlab/graal-match-verifier-review`. Le dépôt GitHub a donc besoin des
   secrets `DOCKERHUB_USERNAME` et `DOCKERHUB_TOKEN` (token Docker Hub *Read &
   Write* sur cet espace de noms). Pour pousser sur un compte personnel, changer
   `IMAGE:` dans le workflow **et** `image:` dans `deployment.yaml`.

2. **Le secret applicatif.** Le Deployment lit `my-s3-creds` (déjà présent dans le
   namespace) pour S3, et un secret `graal-match-verifier-review` pour le reste :

   ```bash
   kubectl create secret generic graal-match-verifier-review \
     --from-literal=NEO4J_URL=... \
     --from-literal=NEO4J_USERNAME=... \
     --from-literal=NEO4J_PWD=... \
     --from-literal=EMBEDDING_MODEL=... \
     --from-literal=URL_EMBEDDING_API=... \
     --from-literal=OPENAI_API_KEY=...
   ```

   Il est marqué `optional: true` : sans lui l'app démarre quand même, la connexion
   Neo4j échoue proprement et la revue se fait sans les notices de code. Les trois
   dernières variables sont lues par `src/neo4j_graph/graph.py` à la construction
   du `Graph`, même si la revue ne fait pas de recherche vectorielle.

3. **Le run revu doit avoir été produit.** `deployment.yaml` épingle
   `--commit 1cace07 --model qwen3-6-35b-moe`, c'est-à-dire
   `s3://projet-ape/graal/data/eval/match_verifier_eval/1cace07/qwen3-6-35b-moe/match_verifier_eval.parquet`.
   Si ce chemin n'existe pas, l'app refuse de démarrer en listant les runs
   disponibles (visible dans les logs du pod). Pour revoir un autre run, changer ces
   deux arguments — et eux seuls.

## Ce qu'il ne faut pas changer en cours de campagne

`--input`, `--reviewers`, `--shared-n`, `--unique-n` et `--seed` déterminent, de
façon déterministe, quelles lignes vont à quel annotateur. Les modifier alors que la
revue a commencé redistribue les paquets sous les pieds des annotateurs : leur
progression reste dans le JSONL mais ne correspond plus à ce que l'app leur présente.

`replicas: 1` et `strategy: Recreate` ne sont pas non plus décoratifs : les
jugements sont ajoutés à un JSONL unique sur S3, et un append S3 réécrit l'objet
entier. Deux pods concurrents perdraient des jugements. L'app sérialise déjà ses
écritures, mais seulement à l'intérieur d'un processus.

## Accès

L'ingress expose `https://graal-match-verifier-review.lab.sspcloud.fr`, sans
authentification : le nom de l'annotateur est un paramètre d'URL, donc n'importe qui
disposant de l'adresse peut écrire des jugements sous n'importe quel nom. Pour mettre
un basic auth devant, créer le secret htpasswd puis décommenter les trois annotations
`auth-*` de `ingress.yaml` :

```bash
htpasswd -c auth meilame && htpasswd auth theo && htpasswd auth nathan
kubectl create secret generic graal-match-verifier-review-basic-auth --from-file=auth
```

## Suivi

```bash
kubectl logs -f deployment/graal-match-verifier-review
```

Les jugements sont consultables à tout moment sous
`s3://projet-ape/graal/data/eval/human_review/<commit>/<modèle>/` : le JSONL
append-only fait foi, le parquet à côté est réécrit après chaque soumission. La page
`/metrics` de l'app donne la même chose agrégée (accord inter-annotateurs compris).
