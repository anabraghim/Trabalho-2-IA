# Trabalho 2 – Classificação de Notícias com BERT

**Matéria:** Inteligência Artificial  
**Ano:** 2025  
**Tema:** Classificação de comentários de notícias utilizando BERT  

---

## Descrição

Este trabalho tem como objetivo treinar um modelo BERT para classificar comentários em três categorias de sentimento: **positivo**, **neutro** e **negativo**, utilizando o token `[CLS]` como representação do texto. O modelo foi adaptado para processar três rótulos simultaneamente: `onça`, `caseiro` e `notícia`.

O notebook contém todas as etapas do projeto, desde a preparação dos dados até a avaliação final do modelo.

---

## Entregáveis

- **Notebook Jupyter:** [Clique aqui para acessar](/trabalho_2_IA_anabraghim.ipynb)
- **Vídeo de Apresentação:** [Clique aqui para assistir](https://youtu.be/UEIwu_j854k) 

## Estrutura do Notebook

1. **Preparação dos Dados**
   - Leitura do CSV com pandas.
   - Seleção das três classes: `onça`, `caseiro` e `notícia`.
   - Limpeza dos dados (remoção de textos vazios e duplicados).
   - Conversão dos rótulos em números.
   - Divisão em conjuntos de treino (70%), validação (15%) e teste (15%).

2. **Tokenização**
   - Uso do tokenizer `neuralmind/bert-base-portuguese-cased`.
   - Geração de tensores `input_ids` e `attention_mask`.

3. **Modelo**
   - Definição do modelo BERT com uma camada linear para previsão das três classes simultaneamente.
   - Extração do token `[CLS]` como representação do texto.

4. **Treinamento**
   - Otimizador: `AdamW` com learning rate `2e-5`.
   - Número de épocas: 10.
   - Monitoramento do **loss** e da **acurácia** em treino e validação por época.
   - Gráfico da evolução do loss.

5. **Avaliação**
   - Avaliação no conjunto de teste.
   - Métricas de **precision**, **recall** e **F1-score** para cada classe.
   - Exemplos de erros com textos classificados incorretamente.
   - Função de predição para testar novos textos.

---

## Como usar

O notebook está pronto para execução. Basta abrir o arquivo no Jupyter Notebook ou no Google Colab e executar as células na ordem apresentada.

### Testando textos

No final do notebook há uma função `prever_texto(texto, model, tokenizer, device)` que permite testar comentários individualmente. Por exemplo:

```python
texto = "essa onça é boa"
resultado = prever_texto(texto, model, tokenizer, device)
print(resultado)
