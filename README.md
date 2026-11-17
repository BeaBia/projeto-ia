# Prototipo de analise dermatologica por imagem

Aplicacao academica experimental com imagens do PAD-UFES-20. O pipeline combina normalizacao e segmentacao heuristica, caracteristicas geometricas, embeddings da MobileNetV3 Small pre-treinada e um MLP do scikit-learn.

**Nao e um dispositivo medico.** Os resultados nao diagnosticam nem descartam cancer e nao devem orientar condutas. O alvo positivo e BCC/SCC/MEL; os demais diagnosticos formam uma classe agregada que pode incluir queratose actinica, uma condicao pre-cancerosa. Logo, escore abaixo do limiar nao significa benignidade.

## Windows e VS Code

Abra a pasta do projeto no VS Code e use um terminal PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O PyTorch pode exigir uma instalacao diferente conforme CPU, CUDA e versao do Python. Se a instalacao padrao nao corresponder ao equipamento, use o seletor oficial de comandos de instalacao do PyTorch. Na primeira inferencia/treinamento, os pesos ImageNet da MobileNetV3 podem ser baixados pela Torchvision.

Coloque `metadata.csv` e as imagens dentro de `data/`, ou um arquivo zip com ambos. Alternativamente, escreva em `caminho_kaggle.txt` um caminho local absoluto ou relativo a raiz do projeto. O KaggleHub e opcional e tenta baixar `mahdavi1202/skin-cancer` quando nao encontra uma base local; isso requer internet e configuracao de acesso conforme as regras do Kaggle.

Treine sem iniciar a interface:

```powershell
python -m ai.treinamento
```

Para retreinar deliberadamente e sobrescrever os artefatos atuais:

```powershell
python -c "from ai.treinamento import treinar_modelo; treinar_modelo(forcar=True)"
```

Inicie a aplicacao:

```powershell
python -m streamlit run app.py
```

Se nao existir um modelo gerado localmente, a aplicacao informa como treina-lo; ela nunca treina automaticamente ao abrir.

## Estrutura

- `ai/preprocessamento.py`: normalizacao de cor, segmentacao e caracteristicas geometricas.
- `ai/extrator_cnn.py`: MobileNetV3 e extracao de embeddings em CPU/GPU.
- `ai/dataset.py`: descoberta de metadata/imagens e leitura de arquivos/zip.
- `ai/treinamento.py`: particoes por paciente, scaler, balanceamento, MLP e persistencia.
- `ai/avaliacao.py`: limiar F2 da validacao e metricas do teste.
- `ai/predicao.py`: caracteristicas e resultado estruturado de inferencia.
- `services/`: coordenacao da inferencia sem acesso a dados pessoais.
- `pages/`: apresentacao e navegacao da aplicacao.
- `original_app.py`: copia de referencia do script monolitico recebido; nao e usado pela aplicacao.

## Metodologia e limitacoes conhecidas

A implementacao preserva a divisao aninhada com `StratifiedGroupKFold` (paciente separado), o limite de amostras por grupo, `StandardScaler`, balanceamento por reamostragem, MLP `(128, 32)`, `alpha=0.10`, `early_stopping=True`, `max_iter=400`, MobileNetV3 Small e busca do limiar F2 entre 0.25 e 0.65 na validacao. O conjunto de teste permanece reservado para metricas e nao seleciona limiar.

Antes de qualquer mudanca metodologica, ficam registrados estes pontos para avaliacao da equipe:

- O `early_stopping` do MLP cria internamente uma particao aleatoria por imagem, nao por paciente; imagens do mesmo paciente podem aparecer nos lados do ajuste interno. As particoes externas continuam agrupadas por paciente. Uma correcao exige rever a estrategia de parada/validacao do MLP.
- O grupo negativo agrega diagnosticos heterogeneos, inclusive queratose actinica. Deve-se avaliar uma tarefa multiclasses ou politica explicita para condicoes pre-cancerosas antes de interpretar o grupo como benigno.
- `predict_proba` do MLP nao foi calibrado; o escore exibido nao e uma probabilidade clinica. Calibracao e limiar precisam ser avaliados apenas com dados de validacao apropriados.
- A opcao de augmentation continua desativada. Se ativada no futuro, as caracteristicas geometricas sao copiadas da imagem original nas transformacoes, sem recomputacao; isso deve ser corrigido e validado antes de usar augmentation.
- Quando uma classe falta no teste, ROC-AUC e reportada como indisponivel; demais metricas e matriz seguem definidas. Divisoes impossiveis ou classe ausente no treino interrompem o processo com erro.
- Imagens ilegiveis agora interrompem o treinamento em vez de serem substituidas silenciosamente por vetores zerados.

## Artefatos

Modelo e manifesto ficam em `models/`. O manifesto confere o hash SHA-256 e identifica o fluxo produtor; isso detecta arquivo ausente/alterado sem manifesto consistente, mas **nao prova autoria contra um atacante capaz de substituir ambos os arquivos**. Como pickle pode executar codigo ao desserializar, nunca use artefatos recebidos de terceiros. Gere seus artefatos no proprio ambiente de confianca.

Esta versao nao implementa banco de dados, cadastro real, historico persistente, autenticacao ou armazenamento de imagens enviadas.
