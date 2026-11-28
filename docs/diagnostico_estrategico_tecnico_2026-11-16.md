# Episense — diagnóstico técnico e estratégia de evolução

Data de referência: 16/11/2026, America/Campo_Grande.

## 1. Parecer executivo

O Episense é um protótipo funcional de previsão semanal de dengue em Campo Grande, com pesquisa e engenharia substanciais. Ainda não há evidência suficiente para apresentá-lo como produto validado para todo o MS ou para três arboviroses. O próximo marco recomendado é um piloto supervisionado, com previsões prospectivas auditáveis e uma decisão operacional claramente definida.

O principal gargalo imediato não é escolher um algoritmo mais sofisticado: é garantir que a previsão exibida, a validação, a versão do modelo e os dados disponíveis na data de emissão sejam coerentes e rastreáveis. Isso aumenta simultaneamente confiabilidade, capacidade de contratação e valor comercial.

## 2. Escopo e limites da análise

- Inventário de toda a árvore de trabalho, excluindo `.git`, `.venv` e caches: 2.031 arquivos, incluindo 25 arquivos Python, 1.605 arquivos pickle, 90 JSON e 221 CSV. A maior parte são modelos e experimentos, não módulos de produto.
- Inspeção dos fluxos centrais de coleta, processamento, features, treinamento, inferência, API, dashboard, segurança, publicação e documentação; amostragem dos logs e artefatos experimentais.
- Parsing sintático dos 25 arquivos Python: sem erro.
- Inspeção estrutural da base, métricas, listas de features, calibração e arquivo de previsões.
- Inferência local com os 200 modelos atuais: oito horizontes, quantis finitos, não negativos e ordenados na última linha da base.
- Execução de `scripts/validate_final_pipeline.py` no ambiente `.venv` e reprodução isolada de uma falha de promoção com mock e diretório temporário.
- Não houve retreino, alteração dos modelos, atualização de dados, subida dos serviços ou teste de carga. Não foi feita auditoria exaustiva de cada binário, pentest nem validação epidemiológica externa. Não foi lido o conteúdo de `.env`.
- A árvore já continha alterações locais e arquivos não versionados. Eles foram preservados. O único entregável de produto desta análise é este documento.
- O Python global não possui NumPy; o `.venv` existente permite executar os testes realizados. Isso não comprova reprodução de um build Docker limpo.

## 3. Contexto informado pelo fundador

Segundo o relato, o projeto nasceu no 2º ano do ensino médio do Colégio Status, escola privada. O desenvolvimento do software foi feito pelo fundador; os dois colegas contribuíram com diário de bordo e apresentação em feira; a professora intermediou o contato com a SES. Não existem instrumentos sobre desenvolvimento, titularidade ou participação, apenas documentos das feiras com os nomes dos participantes.

A manifestação da SES ocorreu em reunião presencial, sem documento de compromisso. Portanto, o estado comercial deve ser descrito como **interesse técnico verbal**, sem afirmar parceria formal, contratação aprovada ou validação institucional. A atribuição da interrupção ao período eleitoral é uma hipótese relatada, não uma causa verificada nesta análise.

Registro de software antigo no INPI, ausência de registro de marca, intenção de abrir CNPJ, contato com Jeff/Jera e apoio de advogada de propriedade intelectual também foram informados pelo fundador; certificados e contratos não foram apresentados nem verificados.

## 4. Mapa do sistema

| Componente | Implementação observada | Implicação |
|---|---|---|
| Fontes | InfoDengue e histórico meteorológico Open-Meteo | Dependência de disponibilidade, revisões e condições de uso externas |
| Coleta | Scripts offline e coleta também dentro dos serviços | Lógica duplicada e múltiplos possíveis escritores |
| Base | CSV semanal, 827 linhas, SE 201101–202644 | Uma série municipal; não é base estadual |
| Features | Lags, médias móveis, sazonalidade, clima, tendência, regime e fase | Engenharia extensa para uma série relativamente pequena |
| Modelo | LightGBM, 8 horizontes × 5 quantis × 5 seeds = 200 modelos | README ainda descreve 40 modelos e 300 features/horizonte |
| Features selecionadas | Listas por horizonte e quantil, entre 10 e 120 entradas | Necessário versionar junto com cada conjunto de modelos |
| Validação | Walk-forward por temporadas, gap temporal e separação interna para early stopping | Boa base metodológica; revisões dos dados e pós-processamento ainda exigem auditoria |
| Publicação | Correções de nível, perfil das bandas, conformal e limites de q95 | O produto publicado é a cadeia inteira, não apenas o LightGBM |
| API | FastAPI, autenticação por token, cache e recarga por mtime | Funcional para um município; não há isolamento por cliente |
| Dashboard | Outro serviço FastAPI, HTML/Chart.js, motor próprio de inferência e retreino | Não é apenas visualização da API; amplia risco de divergência |
| Persistência | CSV/JSON/pickle em volumes compartilhados | Precisa de transações e versões imutáveis para operação institucional |
| Implantação | Docker Compose, API e dashboard | Ainda não demonstra alta disponibilidade, recuperação testada ou SLA |

A base examinada não contém duplicatas de SE, casos vazios ou semanas marcadas como imputadas. As colunas inspecionadas são agregados epidemiológicos e climáticos; isso não equivale a auditoria de privacidade de toda futura integração. Não confundir a presença de `casos` com comprovação de diagnóstico confirmado: o dicionário de dados e a definição do alvo precisam ser pactuados com a fonte e a SES.

## 5. Métricas existentes: o que permitem afirmar

Valores lidos de `models/validation_results.json` e `models/metrics_dataframe.csv`, agregados em `ensemble_avg`; não são uma nova validação prospectiva:

| Horizonte | MAE, casos | WAPE agregado | Cobertura nominal 90% | Cobertura nominal 50% |
|---|---:|---:|---:|---:|
| 1 semana | 38,60 | 21,93% | 86,09% | 52,57% |
| 4 semanas | 65,47 | 43,18% | 88,46% | 54,43% |
| 8 semanas | 90,12 | 68,52% | 88,46% | 34,81% |

O WAPE agregado salvo resulta da agregação dos folds; não deve ser apresentado automaticamente como erro percentual de cada previsão nem convertido em “acurácia = 100% − WAPE”. Cobertura também não é acerto pontual.

As médias escondem heterogeneidade: h1/fold0 tem cobertura90 de 65,38%; h8/fold3, 69,23%. O desempenho relativo ao baseline sazonal é favorável nas médias salvas, mas isso não demonstra superioridade frente à persistência, ao processo atual da SES ou a outros sistemas.

Na reexecução de `validate_final_pipeline.py`, h5 apresentou cobertura50 de aproximadamente 33,0% e cobertura90 de 79,9% na variante expanding. A variante com pool de produção teve resultados melhores, mas reutiliza no ajuste dados dos folds avaliados: não é evidência independente de generalização. Mesmo a variante expanding conserva um perfil fixo selecionado com base no histórico de avaliação; é necessário separar também a seleção desse perfil.

## 6. Achados prioritários

### P0 — histórico exibido não equivale a previsão publicada no passado

Evidência: `dashboard/app.py`, funções `_reproject_archive_forecasts`, `api_history` e `lookup_validation`. O motor atual recalcula origens antigas; o campo `forecast` é substituído, e o original é guardado em `published_forecast`. As 27 entradas do arquivo local tinham sido reprojetadas e possuíam esse campo preservado. O histórico prioriza reprojeções nos pontos cobertos. Há também aplicação de calibração do pool completo sobre previsões de validação.

Cortar a entrada em `SE <= origem` não elimina conhecimento futuro incorporado nos pesos do modelo, na seleção de parâmetros ou nas revisões retrospectivas dos dados. Portanto, a descrição “honesto por construção” no comentário não é suficiente.

Correção proposta: separar explicitamente três produtos: previsão emitida e imutável; backtest temporal; simulação retrospectiva com modelo atual. Usar a primeira para desempenho prospectivo. Recuperar `published_forecast` com indicação de procedência; verificar quais registros foram de fato emitidos na data e quais foram preenchidos posteriormente. Não presumir que o campo, sozinho, certifica emissão histórica.

Critério de aceite: retreinar não altera uma previsão emitida; cada previsão registra origem, emissão, versão dos dados, hash do modelo e configuração; simulações não entram na métrica prospectiva.

### P0 — validação e inferência não compartilham toda a cadeia

Evidência: `models/inference.py::predict`, `scripts/recompute_metrics.py`, `scripts/validate_final_pipeline.py` e configuração. A inferência aplica `median_bias_factor`, `band_shape` e `phantom_gating`; o recálculo canônico não reproduz todas essas etapas, e `phantom_parity` está desativado. O script chamado de validação final explicitamente omite o gating. Nesse script, a mediana usada em MAE fica sem o fator aplicado às bandas. Sua métrica `wis` soma pinball, enquanto `models/metrics.py` usa média: valores absolutos não são diretamente comparáveis.

Correção proposta: uma função única, sem efeitos colaterais, para transformação e publicação; ajuste de cada parâmetro só no passado de cada fold; mesmos cálculos de métricas em todos os caminhos. Avaliar a cadeia final, inclusive limites das bandas, e reservar dados intocados para decisão final.

Critério de aceite: teste de paridade numérica entre backtest e inferência para a mesma entrada e os mesmos artefatos; resultados por horizonte, temporada e regime. Um gráfico visualmente uniforme não justifica redução arbitrária de incerteza.

### P0 — promoção aceita falha do recálculo

Evidência: `scripts/retrain_pipeline.py::finalize_retrain` define `promoted=True` depois de `recompute_canonical`, sem exigir sucesso dos resultados. Em reprodução isolada com gate aprovado e retornos `conformal=False`/`metrics=False`, o resultado foi `promoted=True`, `rollback=False`.

Correção proposta: reprovar se qualquer etapa obrigatória falhar, se o gate estiver inválido ou sem cobertura suficiente; exigir artefatos completos e coerentes.

### P0 — novos modelos são escritos no diretório observado pela API

Evidência: treino grava em `models/`; API recarrega por mtime; Compose compartilha esse diretório com o dashboard. O backup e rollback existentes são úteis, mas não constituem uma promoção atômica. Há uma janela potencial para carregar um conjunto parcialmente atualizado antes da aprovação. Esta condição foi identificada no desenho, não reproduzida sob concorrência.

Correção proposta: treinar em diretório de versão isolado, validar todo o pacote e trocar um único apontador ativo atomicamente. Um coordenador de jobs deve controlar coleta/retreino; locks locais de thread não coordenam processos e containers.

### P1 — dados de treino e dados realmente disponíveis na emissão diferem

Evidência: `data/process_data.py` usa `casos`; `data/live_utils.py::apply_nowcast` substitui casos recentes por `casos_est` quando maior, por até 16 semanas. O coletor atualiza a série acumulada com revisões. Existem CSVs de coletas datadas, mas não foi demonstrada uma reconstrução completa da informação disponível em cada origem.

Correção proposta: guardar snapshots com horário de coleta e semana de referência; avaliar a política de nowcast usando o que estava disponível na emissão. Tratar atraso, revisão e indisponibilidade como variáveis explícitas. Lacuna não deve automaticamente representar zero casos.

### P1 — expansão ainda requer arquitetura e validação próprias

Evidência: `api/main.py` rejeita códigos diferentes do configurado, mantém `disease=dengue` na URL, usa coordenadas únicas e caminhos `infodengue_cg_*`. Alertas usam cortes absolutos 10/50/100.

Correção proposta: chave município × doença × semana, registro de população/coordenadas/fonte, partição de artefatos, configuração de alertas por contexto e validação espacial. Comparar contagens e incidência, sem transportar automaticamente os limites de Campo Grande para municípios pequenos.

### P1 — operação e governança de acesso incompletas

Pontos positivos: token obrigatório em produção, comparação de segredo em tempo constante, CORS explícito, rate limiting e exclusão do `.env` do contexto Docker. O `.env` não está rastreado no estado Git consultado; não houve varredura completa de segredos em todo o histórico.

Lacunas: token compartilhado sem identidade individual e segregação de funções; rate limiting por processo; ausência de TLS no Compose examinado; dashboard executado como root no Dockerfile; retreino dentro do serviço web; persistência compartilhada sem banco transacional. Proxy HTTPS externo pode existir, mas não foi verificado. Os argumentos de SHA no Dockerfile não são utilizados em uma verificação efetiva durante a instalação.

Proposta: usuários e papéis, registro de auditoria, TLS, backups com restauração testada, gestão de segredos, jobs isolados, monitoramento de atraso dos dados e comportamento explícito quando uma fonte falha. Testar prontidão dos modelos separadamente da simples disponibilidade HTTP.

### P1 — baseline e gate precisam representar a decisão real

O gate aceita MAE até 1,35 vezes o baseline sazonal em folds completos. Isso não exige superioridade e não verifica sozinho cobertura, picos, custo de falso alerta ou vantagem sobre persistência. Acrescentar baseline da última observação, média recente, sazonal e referência operacional pactuada. Avaliar semanas de crise separadamente; não escolher critérios depois de ver o resultado.

### P2 — documentação e organização de experimentos

README, relatório histórico, comentários e artefatos descrevem estados distintos. As pastas `models_q*` e `models_v8phase*` são experimentos, não prova de modelos para outros municípios. Separar releases de experimentos, criar manifesto de execução e atualizar documentação a partir da versão publicada. Não apagar esses experimentos sem preservar sua procedência.

## 7. Estratégia de produto e diferenciação

Hipótese de proposta de valor: “Apoiar a vigilância de arboviroses na priorização de municípios e no planejamento semanal de recursos, com previsão auditável e incerteza explícita.” Isso precisa ser validado com usuários e compradores, não apenas aceito em reunião.

O InfoDengue já oferece alerta precoce e visualizações municipais e estaduais, além de fornecer dados ao Episense. Portanto, “IA que prevê dengue em um dashboard” é diferenciação insuficiente. A própria fonte descreve sua atuação nacional e seus produtos: [InfoDengue — informações](https://info.dengue.mat.br/informacoes/).

Escolher uma primeira decisão concreta com a SES: priorização de investigação, planejamento de insumos ou organização de equipes, com horizonte e rotina definidos pelos responsáveis. Não prometer redução de casos ou economia causal apenas porque MAE melhorou.

Medir no piloto: antecedência útil; sensibilidade e falsos alertas segundo definição acordada; cobertura dos intervalos; atraso dos dados; tempo poupado na rotina; frequência de uso; decisões registradas e custo de operação. O piloto deve ter responsável técnico, acesso a dados, escopo, duração, critérios de aceite e reunião final com o decisor.

## 8. Expansão para MS, chikungunya e zika

Sequência recomendada:

1. Corrigir rastreabilidade e validação em Campo Grande/dengue.
2. Selecionar, com dados e equipe técnica, um pequeno grupo de municípios representando portes, regimes e qualidade de notificação distintos; por exemplo, 3–5 municípios, sem afirmar que esse número valida o estado inteiro.
3. Comparar modelos locais e modelos que aprendam conjuntamente entre municípios; usar validação temporal e exclusão de municípios/regiões no teste. Não multiplicar automaticamente os 200 modelos por cada combinação territorial.
4. Adicionar chikungunya como experimento separado, com dicionário de dados, disponibilidade histórica, limiares e métricas próprios.
5. Avaliar zika conforme volume de informação. Séries esparsas podem demandar previsão de ocorrência/risco, modelos de contagem ou agregação, em vez de aparentar precisão em contagens semanais.
6. Ampliar cobertura quando houver critérios mínimos satisfeitos e um procedimento de abstenção: “dados insuficientes” é uma saída válida.

A API do InfoDengue aceita dengue, chikungunya e zika como parâmetro. Isso demonstra possibilidade de consulta, não completude ou qualidade uniforme em todos os municípios: [documentação oficial](https://info.dengue.mat.br/services/api/doc).

Arquitetura-alvo inicial: armazenamento de snapshots, tabelas estruturadas de observação/previsão, pipeline compartilhado, jobs fora do servidor web, catálogo versionado de modelos e API parametrizada. Não é necessário começar com uma arquitetura distribuída complexa.

## 9. Proteção do ativo e organização da empresa

Esta seção é uma pauta para a advogada e o contador, não conclusão sobre direitos de pessoas que não foram ouvidas nem sobre documentos que não foram examinados.

- **Autoria e participação:** documentar contribuições efetivas, cronologia, materiais da feira, código e conversas. Autoria do software, autoria do material acadêmico e participação societária são questões distintas. Não presumir divisão igual por nomes na feira, nem exclusividade jurídica apenas pelo relato de quem programou. Verificar termos da escola/feiras, bolsas e eventuais vínculos.
- **Colegas e professora:** propor esclarecimento escrito das contribuições e dos direitos aplicáveis, redigido pela advogada e coerente com os fatos. Não alterar retroativamente o diário de bordo ou apagar créditos. Relações pessoais não substituem regras de decisão, saída e propriedade.
- **Menoridade:** como o projeto nasceu no ensino médio, confirmar idades e capacidade para contratar, representar a empresa e assinar instrumentos. O portal de abertura de empresas de MS prevê documentação para menores emancipados, representados ou assistidos; o enquadramento específico deve ser examinado: [serviço oficial de MS](https://www.ms.gov.br/servico/registrar-abertura-de-empresa14).
- **Registro de software:** preservar exatamente o pacote que gerou o hash antigo, certificado, recibos e histórico de versões. Avaliar registro de versão atual com a advogada. O INPI descreve o registro como apoio à comprovação de autoria/titularidade, e a proteção se refere ao software, não a exclusividade sobre toda ideia de previsão epidemiológica: [guia do INPI](https://www.gov.br/inpi/pt-br/servicos/programas-de-computador/guia-basico), [serviço de registro](https://www.gov.br/pt-br/servicos/solicitar-o-registro-de-programa-de-computador).
- **Marca:** realizar busca de anterioridades e estratégia de depósito antes de investimento relevante na identidade. Não foi feita busca de disponibilidade de “Episense”.
- **NDA:** útil para limitar uso/divulgação de informações definidas, mas não transfere propriedade nem resolve autoria, contratação ou exclusividade. Identificar partes, finalidade, informações protegidas, exceções e registro das entregas. Não presumir que um agente público possa comprometer o órgão mediante instrumento genérico.
- **Governo:** compatibilizar sigilo com regras de transparência e hipóteses legais de proteção; não assumir sigilo absoluto de toda apresentação. A orientação pública distingue informação classificada de outras hipóteses legais de sigilo: [GSI — LAI](https://www.gov.br/gsi/pt-br/seguranca-da-informacao-e-cibernetica/perguntas-frequentes-1/perguntas-sobre-tratamento-da-informacao-classificada-e-credenciamento-de-seguranca-lai/).
- **Contratos futuros:** definir tecnologia preexistente, melhorias, licenciamento, direitos sobre dados, publicações, suporte, término e eventual acesso ao código. A Lei de Software torna os vínculos contratuais relevantes à titularidade: [Lei 9.609, texto atualizado](https://www2.camara.leg.br/legin/fed/lei/1998/lei-9609-19-fevereiro-1998-364738-normaatualizada-pl.html).
- **Divulgação por etapas:** apresentação inicial com problema, resultados e limitações; demonstração controlada; documentação técnica sob finalidade acordada. Repositório, pesos, credenciais e cópia integral do ambiente não são necessários para a primeira conversa.
- **CNPJ:** avaliar forma societária, atividade, tributação, elegibilidade a editais e transferência/licença do ativo para a empresa. Não assumir que o software pertence automaticamente ao CNPJ criado depois.

O risco de apropriação deve ser tratado com evidências, contratos e controle de acesso. Não há, no material examinado, evidência de tentativa de apropriação pela SES. A indefinição interna e a entrega sem termos são riscos concretos que podem ser reduzidos agora.

## 10. Financiamento, contratação e relacionamento

| Caminho | Papel | Próxima evidência necessária |
|---|---|---|
| Centelha/Fundect | Apoio condicionado a edital e elegibilidade | Chamada, prazo, proponente, estágio, despesas permitidas e contrapartidas |
| Finep | Instrumentos distintos, conforme programa | Separar subvenção, crédito e outros mecanismos; verificar chamada e obrigações |
| Piloto institucional | Testar utilidade e reduzir risco | Plano de trabalho, responsável, recursos, dados e critérios de aceite |
| Contrato público | Compra formal | Demanda, orçamento, enquadramento jurídico e procedimento competitivo aplicável |
| Cliente privado | Receita e aprendizado | Quem decide, que decisão muda, disposição a pagar e dados necessários |
| Parceiro tecnológico | Entregas e capacidade | Escopo, remuneração, PI, confidencialidade e marcos |

Em 16/11/2026, a consulta da Fundect registra resultado final da Fase 2 do Centelha em 29/10/2026. Não tratar essa edição como inscrição inicial aberta. Não foi identificada nesta análise uma chamada aberta e elegível já confirmada para o Episense: [consulta oficial](https://www.fundect.ms.gov.br/informativos/consultas/).

“Finep” e “Fundect” não são um único cheque disponível; é preciso escolher um instrumento concreto. A Finep orienta consultar seus programas e chamadas: [atendimento oficial](https://faleconosco.finep.gov.br/sac). Não recomendar dívida sem fluxo de caixa e capacidade de pagamento demonstráveis.

Uma via a discutir com o jurídico da SES é a contratação de teste de solução inovadora no marco legal das startups, quando houver adequação ao problema e procedimento próprio. A LC 182 prevê licitação especial para esses testes; isso não significa contratação garantida ou dispensa automática: [artigo 13](https://normas.leg.br/?urn=urn%3Alex%3Abr%3Afederal%3Alei.complementar%3A2021-06-01%3B182%21art13).

Na retomada com a SES, buscar confirmação escrita do interesse, interlocutor técnico, problema prioritário, decisor e caminho para piloto. Não usar nome/logo da SES como endosso comercial sem autorização. A manifestação técnica e a decisão de compra podem pertencer a pessoas diferentes.

Com Jeff, definir primeiro o papel pretendido: mentor, fornecedor, parceiro de implantação, canal comercial ou investidor. Uma primeira entrega útil seria uma revisão de arquitetura e esforço para um piloto delimitado. Não oferecer participação societária apenas por acesso a contatos; qualquer equity deve corresponder a compromissos, entregas e regras de saída formalizados.

## 11. Modelo econômico inicial

Hipótese a validar: implantação/piloto com escopo definido, seguida de licença ou serviço recorrente com suporte e monitoramento. Venda integral do ativo/empresa é uma alternativa futura; sem titularidade organizada, validação e receita, há pouca base para estimar valor de aquisição.

Separar: receita contratada, receita recebida, recurso de edital sujeito a obrigações e caixa livre. Interesse verbal não entra como receita. Trabalho do fundador deve entrar no custo, mesmo quando não há retirada financeira imediata.

Orçamento mínimo deve incluir engenharia, validação epidemiológica, infraestrutura, fontes de dados, suporte, jurídico/contábil, segurança, impostos e capital de giro. Construir cenários de atraso comercial e de recebimento; não definir preço somente pelo custo do servidor.

Fórmulas gerenciais iniciais:

- Caixa necessário = implantação + despesas mensais × meses até recebimento + reserva − caixa disponível − entradas confirmadas.
- Margem de contribuição = receita líquida de tributos − custos variáveis de entrega e suporte.
- Ponto de equilíbrio em clientes = custo fixo mensal / contribuição mensal por cliente, apenas se os contratos forem comparáveis.

Não há informação suficiente nesta pasta para estimar valuation, preço de venda, investimento necessário ou margem real. A primeira proposta deve ser orçada a partir do esforço e validada com disposição a pagar.

Dependência comercial específica: a API gratuita do Open-Meteo é restrita a uso não comercial, enquanto a licença dos dados e atribuição são questões relacionadas, mas distintas. Prever plano comercial ou alternativa compatível antes da comercialização: [termos oficiais](https://open-meteo.com/en/terms). Verificar também condições de redistribuição das demais fontes e licenças das dependências; não houve auditoria jurídica completa dessas licenças.

## 12. Plano de execução sugerido

As janelas abaixo são metas de organização, não estimativa fechada de esforço nem garantia de validação sazonal em 90 dias.

| Janela | Entrega | Critério de conclusão |
|---|---|---|
| Dias 1–15 | Separar histórico publicado/backtest/simulação; corrigir promoção; documentar contribuições | Previsão emitida imutável; falha de etapa bloqueia publicação; pauta jurídica com evidências |
| Dias 16–30 | Pipeline único, manifesto de versão, baselines e plano de piloto | Paridade testada; protocolo pré-definido; interlocutor e decisão operacional identificados |
| Dias 31–60 | Piloto supervisionado e coleta versionada; parametrização de municípios | Previsões emitidas antes dos desfechos; qualidade de dados monitorada; operação rastreável |
| Dias 61–90 | Avaliação inicial, pequeno conjunto territorial e orçamento de expansão | Relatório técnico-comercial com limitações, custos e decisão de continuidade |

Uma janela curta sem surto não comprova desempenho em surto. A validação prospectiva deve continuar pelo período necessário, acompanhada de avaliação histórica temporalmente correta.

## 13. Próxima intervenção recomendada

Começar pela integridade da evidência: preservar e exibir previsões efetivamente publicadas, separar simulações, unificar inferência e validação e impedir publicação incompleta. Depois, executar um benchmark reproduzível com baselines fortes. Essa sequência evita otimizar um número que não representa o produto entregue.

Em paralelo, a advogada organiza titularidade, capacidade contratual, marca e instrumentos de relacionamento; o fundador transforma o interesse verbal da SES em escopo de piloto documentado. Melhorar o modelo, estruturar a empresa e validar a compra são frentes diferentes, com entregas próprias.
