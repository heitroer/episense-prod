# Wolbachia e previsão de dengue em Campo Grande

O estudo ecológico local comparou incidência mensal por bairro e encontrou redução estimada de 63,2% (IC 95% de 51,9% a 71,9%) nas áreas com prevalência de wMel estável de pelo menos 60%. Em 2024, 89% das áreas alcançaram esse critério. O efeito varia conforme o estabelecimento local. Aplicar 63,2% de desconto às previsões municipais também pode contar o efeito duas vezes, pois a série usada pelo modelo já incorpora a intervenção. [Artigo original](https://pmc.ncbi.nlm.nih.gov/articles/PMC12719740/)

O atributo opcional `wolbachia_cg_stage` registra o último marco público do programa na origem da previsão. Vale zero antes da SE 202051, um a partir da SE 202051 e dois a partir da SE 202430. Cada transição começa no domingo seguinte à publicação do comunicado. A última categoria identifica o encerramento divulgado das solturas. A prevalência nos anos seguintes permanece uma incógnita. [Início divulgado pela OPAS em dezembro de 2020](https://www.paho.org/pt/noticias/10-12-2020-municipio-campo-grande-inicia-uso-do-metodo-wolbachia-contra-zika-dengue-e) · [Encerramento divulgado pela OPAS em julho de 2024](https://www.paho.org/pt/noticias/18-7-2024-wolbachia-contra-dengue-opas-faz-acompanhamento-tecnico-do-inicio-ao)

## Reprodução

Execute na raiz com o ambiente do projeto e um diretório novo sob logs.

```sh
.venv/bin/python scripts/benchmark_model_variants.py --output logs/comparacao_wolbachia --variant full_one_seed --variant full_wolbachia_one_seed --all-horizons --revised
```

O teste usa uma semente comum, a mesma base revisada, cinco quantis e os mesmos cortes temporais. O embargo respeita a maturação dos alvos. A avaliação de desenvolvimento mede a contribuição preditiva do indicador e não estima causalidade nem reproduz as revisões disponíveis em cada emissão histórica.
