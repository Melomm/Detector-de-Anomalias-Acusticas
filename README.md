# Detector de Anomalias Acústicas - “6 7”

Sistema embarcado que identifica as sequências **“seis sete”** e **“six seven”** com ESP32-WROOM-32U, microfone INMP441 e FreeRTOS. A decisão ocorre no ESP32 e aciona um LED. O padrão escolhido representa um comando vocal curto para acionamento local, sem necessidade de conexão com a internet na placa física.

O sistema diferencia a sequência completa dos exemplos negativos usados no treinamento: números isolados, ordem invertida, outras falas e ruídos. Trata-se de classificação supervisionada de um padrão acústico, não de reconhecimento geral de fala nem de identificação irrestrita de anomalias.

![Arquitetura de tarefas e sincronização](docs/rtos.svg)

## Entregáveis

| Item solicitado no enunciado | Arquivo ou diretório |
|---|---|
| Código-fonte embarcado | [firmware/src/main.cpp](firmware/src/main.cpp) e [firmware/include](firmware/include) |
| Diagrama RTOS, primitivas e sincronização | [docs/rtos.svg](docs/rtos.svg) |
| Modelo de detecção ONNX | [modelo/model.onnx](modelo/model.onnx) |
| Relatório técnico: arquitetura, latência, resultados e discussão | [docs/relatorio-tecnico.md](docs/relatorio-tecnico.md) |
| Script de simulação de eventos e desempenho | [scripts/simulate.py](scripts/simulate.py) |
| Análise de latência medida no ESP32 | [scripts/summarize_latency.py](scripts/summarize_latency.py) |
| Evidências e resultados numéricos | [docs/evidencias](docs/evidencias) |

## Resultados registrados

- **Modelo:** 74 gravações; 46 para treino, 14 para validação e 14 para teste. Precisão, recall e F1 de **83,3%** no teste por clipe.
- **Placa física:** 65 janelas consecutivas, dois eventos emitidos e nenhum erro de leitura ou descarte no trecho fornecido pelo autor.
- **Latência de processamento na placa:** média de **185,999 ms**, máximo observado de **186,043 ms**, com atualização a cada **250 ms**. Esse tempo não representa a duração entre o início da fala e o alerta.
- **Verificação automatizada:** 19 testes aprovados. O relatório distingue testes com sinais sintéticos, avaliação de gravações e observação no hardware.

## Organização

`src/detector67` contém preparação de dados, DSP, treinamento, exportação e avaliação em Python. `firmware` contém captura I2S, processamento e inferência C++ com as três tarefas FreeRTOS. `audios/positivo`, `audios/negativo` e `audios/teste` recebem os dados locais. Os áudios e os artefatos de trabalho em `artifacts` não são versionados.

`modelo` contém uma cópia da versão avaliada: ONNX, pesos NumPy, metadados e cabeçalho C++. Essa cópia permite entregar o modelo mesmo com `artifacts` ignorado pelo Git. Novos treinos não substituem automaticamente a versão documentada.

Os comandos de entrada são `treinar.cmd`, `testar.cmd`, `compilar_modelo.cmd`, `microfone.cmd` e `testar_microfone.cmd`. `compilar_modelo.cmd` incorpora os pesos de um modelo já treinado ao firmware e compila o ambiente `detector`, sem fazer upload. A simulação Wokwi possui modos de demonstração sintética, reprodução de arquivos e microfone do PC. A validação física utiliza o INMP441; a simulação não comprova suas conexões elétricas.

Requisitos e correspondência com a implementação: [enunciado original](<Detector de Anomalias Acústicas.pdf>) e [relatório técnico](docs/relatorio-tecnico.md).
