# Databricks notebook source
# MAGIC %md
# MAGIC ### Projeto 1 - Segmentação de Clientes de Food Delivery

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marketing Analytics
# MAGIC
# MAGIC Marketing Analytics compreende os processos e tecnologias que permitem aos profissionais de Marketing avaliar o sucesso de suas iniciativas.
# MAGIC
# MAGIC Isso é feito medindo o desempenho das campanhas de Marketing, coletando os dados e analisando os resultados. Marketing Analytics utiliza métricas importantes de negócios, como ROI (Retorno Sobre o Investimento), Atribuição de Marketing e Eficácia Geral do Marketing. Em outras palavras, o Marketing Analytics mostra se os programas de Marketing estão sendo efetivos ou não.
# MAGIC
# MAGIC Marketing Analytics reúne dados de todos os canais de marketing e os consolida em uma visão de marketing comum. A partir dessa visão comum, você pode extrair resultados analíticos que podem fornecer assistência inestimável para impulsionar os esforços de marketing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Por Que Marketing Analytics é Importante?
# MAGIC
# MAGIC Leia o manual em pdf no próximo item de aprendizagem!

# COMMAND ----------

# MAGIC %md
# MAGIC ### O Que Você Pode Fazer com Marketing Analytics?
# MAGIC
# MAGIC Com Marketing Analytics, você pode responder a perguntas como estas:
# MAGIC
# MAGIC Como estão as nossas iniciativas de marketing hoje? Que tal a longo prazo? O que podemos fazer para melhorá-las?
# MAGIC Como nossas atividades de marketing se comparam às de nossos concorrentes? Onde eles estão gastando seu tempo e dinheiro? Eles estão usando canais que não estamos usando?
# MAGIC O que devemos fazer em seguida? Nossos recursos de marketing estão alocados corretamente? Estamos dedicando tempo e dinheiro aos canais certos? Como devemos priorizar nossos investimentos para o próximo ano?
# MAGIC Qual o perfil dos nossos clientes? Eles são da mesma área regional? Tem os mesmos gostos e preferências?
# MAGIC Consigo segmentar meus clientes por similaridade? Tenho como saber os gastos por grupo?
# MAGIC E muitas outras..

# COMMAND ----------

# MAGIC %md
# MAGIC ### O Que é Segmentação de Clientes?
# MAGIC
# MAGIC A segmentação de clientes é o processo de dividir os clientes em grupos com base em características comuns, para que as empresas possam comercializar para cada grupo de forma eficaz e adequada, ou simplesmente compreender o padrão de consumo dos clientes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marketing B2B x Marketing B2C
# MAGIC
# MAGIC No Marketing Business-to-Business (B2B), uma empresa pode segmentar clientes de acordo com uma ampla variedade de fatores, incluindo:
# MAGIC
# MAGIC Indústria
# MAGIC Número de empregados
# MAGIC Produtos comprados anteriormente na empresa
# MAGIC Localização
# MAGIC No Marketing Business-to-Consumer (B2C), as empresas geralmente segmentam os clientes de acordo com dados demográficos e padrões de consumo, tal como:
# MAGIC
# MAGIC Idade
# MAGIC Gênero
# MAGIC Estado civil
# MAGIC Localização (urbana, suburbana, rural)
# MAGIC Estágio da vida (sem filhos, aposentado, etc.)
# MAGIC Produtos comprados
# MAGIC Valor gasto
# MAGIC Horário de consumo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Por Que Segmentar Clientes?
# MAGIC
# MAGIC Leia o manual em pdf no próximo item de aprendizagem!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Como Segmentar Clientes?
# MAGIC
# MAGIC A segmentação de clientes exige que uma empresa colete informações específicas - dados - sobre clientes e analise-as para identificar padrões que podem ser usados para criar segmentos.
# MAGIC
# MAGIC Parte disso pode ser obtida a partir de informações de compra - cargo, geografia, produtos adquiridos, por exemplo. Algumas delas podem ser obtidas da forma como o cliente entrou no seu sistema. Um profissional de marketing que trabalha com uma lista de e-mail de inscrição pode segmentar mensagens de marketing de acordo com a oferta de inscrição que atraiu o cliente, por exemplo. Outras informações, no entanto, incluindo dados demográficos do consumidor, como idade e estado civil, precisarão ser adquiridas de outras maneiras.
# MAGIC
# MAGIC Os métodos típicos de coleta de informações incluem:
# MAGIC
# MAGIC Entrevistas presenciais ou por telefone
# MAGIC Pesquisas
# MAGIC Coleta de informações publicadas sobre categorias de mercado
# MAGIC Grupos de foco
# MAGIC Dados de acessos a sistemas ou apps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Usando Segmentos de Clientes
# MAGIC
# MAGIC Características comuns nos segmentos de clientes podem orientar como uma empresa comercializa segmentos individuais e quais produtos ou serviços ela promove. Uma pequena empresa que vende guitarras feitas à mão, por exemplo, pode decidir promover produtos com preços mais baixos para guitarristas mais jovens e guitarras premium com preços mais altos para músicos mais velhos, com base no conhecimento do segmento que lhes diz que os músicos mais jovens têm menos renda disponível do que seus colegas mais velhos.
# MAGIC
# MAGIC A segmentação de clientes pode ser praticada por todas as empresas, independentemente do tamanho ou setor, e se vendem on-line ou presencialmente. Começa com a coleta e a análise de dados e termina com a atuação nas informações coletadas de maneira apropriada e eficaz, com a entrega das conclusões.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Marketing B2B x Marketing B2C
# MAGIC No Marketing Business-to-Business (B2B), uma empresa pode segmentar clientes de acordo com uma ampla variedade de fatores, incluindo:
# MAGIC
# MAGIC Indústria
# MAGIC Número de empregados
# MAGIC Produtos comprados anteriormente na empresa
# MAGIC Localização
# MAGIC No Marketing Business-to-Consumer (B2C), as empresas geralmente segmentam os clientes de acordo com dados demográficos e padrões de consumo, tal como:
# MAGIC
# MAGIC Idade
# MAGIC Gênero
# MAGIC Estado civil
# MAGIC Localização (urbana, suburbana, rural)
# MAGIC Estágio da vida (sem filhos, aposentado, etc.)
# MAGIC Produtos comprados
# MAGIC Valor gasto
# MAGIC Horário de consumo

# COMMAND ----------

# MAGIC %md
# MAGIC #### Por Que Segmentar Clientes?
# MAGIC Leia o manual em pdf no próximo item de aprendizagem!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Como Segmentar Clientes?
# MAGIC A segmentação de clientes exige que uma empresa colete informações específicas - dados - sobre clientes e analise-as para identificar padrões que podem ser usados para criar segmentos.
# MAGIC
# MAGIC Parte disso pode ser obtida a partir de informações de compra - cargo, geografia, produtos adquiridos, por exemplo. Algumas delas podem ser obtidas da forma como o cliente entrou no seu sistema. Um profissional de marketing que trabalha com uma lista de e-mail de inscrição pode segmentar mensagens de marketing de acordo com a oferta de inscrição que atraiu o cliente, por exemplo. Outras informações, no entanto, incluindo dados demográficos do consumidor, como idade e estado civil, precisarão ser adquiridas de outras maneiras.
# MAGIC
# MAGIC Os métodos típicos de coleta de informações incluem:
# MAGIC
# MAGIC Entrevistas presenciais ou por telefone
# MAGIC Pesquisas
# MAGIC Coleta de informações publicadas sobre categorias de mercado
# MAGIC Grupos de foco
# MAGIC Dados de acessos a sistemas ou apps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Usando Segmentos de Clientes
# MAGIC Características comuns nos segmentos de clientes podem orientar como uma empresa comercializa segmentos individuais e quais produtos ou serviços ela promove. Uma pequena empresa que vende guitarras feitas à mão, por exemplo, pode decidir promover produtos com preços mais baixos para guitarristas mais jovens e guitarras premium com preços mais altos para músicos mais velhos, com base no conhecimento do segmento que lhes diz que os músicos mais jovens têm menos renda disponível do que seus colegas mais velhos.
# MAGIC
# MAGIC A segmentação de clientes pode ser praticada por todas as empresas, independentemente do tamanho ou setor, e se vendem on-line ou presencialmente. Começa com a coleta e a análise de dados e termina com a atuação nas informações coletadas de maneira apropriada e eficaz, com a entrega das conclusões.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Iniciando o Desenvolvimento do Projeto

# COMMAND ----------

!pip install -q -U watermark

# COMMAND ----------

# DBTITLE 1,Carregando os Pacotes
# Imports

# Manipulação e visualização de dados
import time
import sklearn
import datetime
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib as m
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# Formatação dos gráficos
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
%matplotlib inline

# COMMAND ----------

# Versões dos pacotes usados neste notebook
%reload_ext watermark
%watermark -a "Eng. Diego de Souza" --iversions

# COMMAND ----------

'''!pip install -q -U scikit-learn==0.23.1'''

# COMMAND ----------

# DBTITLE 1,Carregando e Compreendendo os Dados
# File location and type
file_location = "/FileStore/tables/carga/dataset.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# COMMAND ----------

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

df_food_delivery = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise Exploratória
# MAGIC
# MAGIC Vamos explorar os dados por diferentes perspectivas e compreender um pouco mais o relacionamento entre as variáveis.

# COMMAND ----------

# Verifica o total de valores únicos por coluna
df_food_delivery.nunique()

# COMMAND ----------

# Tipos de dados
df_food_delivery.dtypes

# COMMAND ----------

df_food_delivery['localidade'] = df_food_delivery['localidade'].astype('int64')

# COMMAND ----------

df_food_delivery['quantidade_item'] = df_food_delivery['quantidade_item'].astype('int64')

# COMMAND ----------

# Resumo das colunas numéricas
df_food_delivery.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Começaremos criando uma tabela que nos fornecerá o número de vezes cada item foi solicitado em cada pedido.

# COMMAND ----------

# Lista para receber o total de pedidos
total_pedidos = []

# COMMAND ----------

# MAGIC %md
# MAGIC Loop para criar a tabela pivot totalizando os itens por transação.

# COMMAND ----------

# MAGIC %%time
# MAGIC
# MAGIC print("\nIniciando o agrupamento para o cálculo do total de pedidos. Seja paciente e aguarde...") # 
# MAGIC
# MAGIC # Extraímos cada id e cada grupo do 'group by' por id_transacao
# MAGIC for k, group in df_food_delivery.groupby('id_transacao'):
# MAGIC     
# MAGIC     # Extraímos cada id e cada grupo do group by por horario_pedido
# MAGIC     for m, n in group.groupby('horario_pedido'):
# MAGIC         
# MAGIC         # Extraímos cada item de cada grupo
# MAGIC         id_transacao = k
# MAGIC         horario_pedido = m
# MAGIC         localidade = n['localidade'].values.tolist()[0]
# MAGIC         bebida = 0
# MAGIC         pizza = 0
# MAGIC         sobremesa = 0
# MAGIC         salada = 0
# MAGIC         n = n.reset_index(drop = True)
# MAGIC         
# MAGIC         # E então contabilizamos os itens pedidos
# MAGIC         for i in range(len(n)):
# MAGIC             item = n.loc[i, 'nome_item']
# MAGIC             num = n.loc[i, 'quantidade_item']
# MAGIC             
# MAGIC             if item == 'bebida':
# MAGIC                 bebida = bebida + num
# MAGIC             
# MAGIC             elif item == 'pizza':
# MAGIC                 pizza = pizza + num
# MAGIC             
# MAGIC             elif item == 'sobremesa':
# MAGIC                 sobremesa = sobremesa + num
# MAGIC             
# MAGIC             elif item == 'salada':
# MAGIC                 salada = salada + num
# MAGIC         
# MAGIC         output = [id_transacao, horario_pedido, localidade, bebida, pizza, sobremesa, salada]
# MAGIC         total_pedidos.append(output)
# MAGIC         
# MAGIC print("\nAgrupamento concluído!")

# COMMAND ----------

# Convertemos a lista para dataframe
df_item_pedidos = pd.DataFrame(total_pedidos)

# COMMAND ----------

# Ajustamos os nomes das colunas
df_item_pedidos.columns = ['id_transacao', 'horario_pedido', 'localidade', 'bebida', 'pizza', 'sobremesa', 'salada']

# COMMAND ----------

# Shape
df_item_pedidos.shape

# COMMAND ----------

# Verifica o total de valores únicos por coluna
df_item_pedidos.nunique()

# COMMAND ----------

# Visualiza os dados originais
df_food_delivery.head()

# COMMAND ----------

# Visualiza o resultado do pivot
df_item_pedidos.head(10)

# COMMAND ----------

# Vamos criar uma tabela pivot com id_transacao, nome_item e quantidade_item
df_pivot = df_food_delivery.pivot_table(index = ['id_transacao'], columns = ['nome_item'], values = 'quantidade_item')

# COMMAND ----------

# Substituímos possíveis valores NA gerados no pivot, por 0 e transformamos o índice em coluna
df_pivot = df_pivot.fillna(0).reset_index()

# COMMAND ----------

# Tipo do objeto
type(df_pivot)

# COMMAND ----------

# Tipos de dados nas colunas
df_pivot.dtypes

# COMMAND ----------

# Nomes das colunas
df_pivot.columns

# COMMAND ----------

# Visualiza os dados
df_pivot.head()

# COMMAND ----------

# Valores únicos
df_pivot.nunique()

# COMMAND ----------

# Shape
df_pivot.shape

# COMMAND ----------

# Describe
df_pivot.describe()

# COMMAND ----------

# Não podemos ter valores nulos
df_pivot.isnull().sum()

# COMMAND ----------

# Vamos incluir a coluna localidade e para fazer o merge precisamos de uma coluna em comum, nesse caso, id_transacao
df_pivot2 = df_pivot.merge(df_food_delivery[['id_transacao', 'localidade']])

# COMMAND ----------

# Visualiza os dados
df_pivot2.head()

# COMMAND ----------

# Shape
df_pivot2.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extraindo Granularidade de Tempo¶
# MAGIC A coluna de horário do pedido tem detalhes como mês, dia e ano. Em algum momento pode ser interessante fazer a segmentação por mês, por exemplo. Vamos então extrair o mês e colocar em uma coluna separada.

# COMMAND ----------

# Visualiza os dados
df_item_pedidos.head(3)

# COMMAND ----------

# Extraímos o mês da coluna horario_pedido e gravamos em uma nova coluna
df_item_pedidos['mes'] = df_item_pedidos['horario_pedido'].apply(lambda x: time.strftime("%m", time.strptime(x,"%Y-%m-%d %H:%M:%S")))


# COMMAND ----------

df_item_pedidos.head(10)

# COMMAND ----------

# Vamos incluir a coluna localidade e para fazer o merge precisamos de uma coluna em comum, nesse caso, id_transacao
df_pivot = df_pivot.merge(df_item_pedidos[['id_transacao', 'mes']])

# COMMAND ----------

# Visualiza o resultado
df_pivot.head(10)

# COMMAND ----------

# Visualiza valores únicos
df_pivot.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ajuste de Índices
# MAGIC
# MAGIC Para segmentar os pedidos dos clientes, precisamos de uma coluna de identificação de cada registro. Não podemos usar id_transacao, pois essa coluna representa um dado válido e além disso não é um valor único, logo não pode ser usado como índice.
# MAGIC
# MAGIC Vamos então criar uma coluna usando o índice atual, o que acha? Vamos checar o índice:

# COMMAND ----------

# Dataset
df_item_pedidos

# COMMAND ----------

# Índice
df_item_pedidos.index

# COMMAND ----------

# Fazemos o reset no índice e gravamos o resultado em outro dataframe
df_item_pedidos_idx = df_item_pedidos.reset_index()

# COMMAND ----------

# Pronto, agora temos uma coluna de ID com valor único para cada registro
df_item_pedidos_idx.head()

# COMMAND ----------

# Dataset
df_item_pedidos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise Descritiva
# MAGIC
# MAGIC ### Distplot dos Atributos Usados Para Segmentação

# COMMAND ----------

# Plot

# Tamanho da figura
plt.figure(1 , figsize = (15 , 6))

# Inicializa o contador
n = 0 

# Loop pelas colunas
for x in ['pizza' , 'sobremesa' , 'salada', 'bebida', 'localidade']:
    n += 1
    plt.subplot(1 , 5 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.distplot(df_item_pedidos[x] , bins = 20)
    plt.title('Distplot de {}'.format(x))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráfico de Total de Pedidos Por Localidade

# COMMAND ----------

# Plot
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'localidade' , data = df_item_pedidos)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regplot dos Atributos Usados Para Segmentação

# COMMAND ----------

# Relação Entre os Atributos

# Tamanho da figura
plt.figure(1 , figsize = (15 , 7))

# Inicializa o contador
n = 0 

# Loop pelos atributos
for x in ['pizza' , 'sobremesa' , 'salada', 'bebida']:
    for y in ['pizza' , 'sobremesa' , 'salada', 'bebida']:
        n += 1
        plt.subplot(4 , 4 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df_item_pedidos)
        plt.ylabel(y)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definindo as Variáveis Para Segmentação
# MAGIC
# MAGIC Vamos remover id_transacao, horario_pedido, localidade e mes para nossas primeiras atividades de clusterização.

# COMMAND ----------

# Filtrando o dataframe por colunas 
df_item_pedidos_idx[['index', 'bebida', 'pizza', 'sobremesa', 'salada']]

# COMMAND ----------

# Vamos gerar um novo dataframe com o slice anterior
df = df_item_pedidos_idx[['index', 'bebida', 'pizza', 'sobremesa', 'salada']]

# COMMAND ----------

# Dataset
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise de Cluster
# MAGIC
# MAGIC ### Métricas de Clusterização - Definição e Interpretação
# MAGIC
# MAGIC Está disponível no manual em pdf no próximo item de aprendizagem.

# COMMAND ----------

# Usaremos duas variáveis
X1 = df[['pizza' , 'sobremesa']].iloc[: , :].values

# COMMAND ----------

# Lista do WCSS
wcss_X1 = []

# COMMAND ----------

# MAGIC %md
# MAGIC #### Segmentação 1 - Encontrando o Valor Ideal de Clusters
# MAGIC
# MAGIC Vamos testar diferentes valores de K (valores de cluster) entre 2 e 10.
# MAGIC
# MAGIC Para a inicialização dos clusters, usamos o algoritmo k-means++ que oferece convergência mais rápida para o resultado final.
# MAGIC
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# COMMAND ----------

# Loop para testar os valores de K
for n in range(2, 11):
    modelo = (KMeans(n_clusters = n,
                     init = 'k-means++', 
                     n_init = 10,
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan'))
    modelo.fit(X1)
    wcss_X1.append(modelo.inertia_)

# COMMAND ----------

# Plot
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , 11) , wcss_X1 , 'o')
plt.plot(np.arange(2 , 11) , wcss_X1 , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('WCSS')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Escolhemos o valor ideal de clusters e criamos o modelo final para a Segmentação 1. Observe no gráfico acima que não há certo ou errado. Poderíamos trabalhar com qualquer valor entre 2 e 10 (não faz sentido criar apenas 1 cluster).
# MAGIC
# MAGIC O gráfico acima é chamado de Curva de Elbow e normalmente usamos o valor com o menor WCSS. Mas isso deve ser alinhado com as necessidade de negócio. Para esse exemplo, não faria sentido usar 10 clusters. Vamos começar com 2 clusters e avaliar e interpretar os resultados.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Segmentação 1 - Construindo e Treinando o Modelo

# COMMAND ----------

# Criação do modelo
modelo_seg1 = KMeans(n_clusters = 2, 
                     init = 'k-means++', 
                     n_init = 10, 
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan')

# COMMAND ----------

# Treinamento do modelo
modelo_seg1.fit(X1)

# COMMAND ----------

# Extração dos labels
labels1 = modelo_seg1.labels_
labels1

# COMMAND ----------

# Extração dos centróides
centroids1 = modelo_seg1.cluster_centers_
centroids1

# COMMAND ----------

# MAGIC %md
# MAGIC Caso queira alterar a combinação de cores dos gráficos, basta alterar a paleta usada. Aqui estão as opções:
# MAGIC
# MAGIC https://matplotlib.org/3.2.0/tutorials/colors/colormaps.html
# MAGIC
# MAGIC Para o Segmento 1 estamos usando plt.cm.Set2.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Segmentação 1 - Visualização e Interpretação dos Segmentos

# COMMAND ----------

# Plot

# Parâmetros do Meshgrid
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = modelo_seg1.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15, 7) )
plt.clf()
Z = Z.reshape(xx.shape)

# Plot da imagem
plt.imshow(Z, 
           interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Set2, 
           aspect = 'auto', 
           origin = 'lower')

# Plot dos pontos de dados
plt.scatter( x = 'pizza', y = 'sobremesa', data = df, c = labels1, s = 200 )
plt.scatter(x = centroids1[: , 0], y =  centroids1[: , 1], s = 300, c = 'red', alpha = 0.5)
plt.xlabel('Pizza')
plt.ylabel('Sobremesa')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretação**: 
# MAGIC
# MAGIC - O ponto vermelho é o centróide de cada cluster (segmento).
# MAGIC
# MAGIC
# MAGIC - No cluster 1 (área em verde) temos os clientes que pediram 0, 1 ou 2 Pizzas. Em todos os casos houve pedido de Sobremesa.
# MAGIC
# MAGIC
# MAGIC - No cluster 2 (área em cinza) estão clientes que pediram 2, 3, 4 ou 5 Pizzas. Perceba que à medida que o pedido tem maior número de Pizzas, também aumenta o número de Sobremesas.
# MAGIC
# MAGIC **Análise**:
# MAGIC
# MAGIC - Cluster 1 - Clientes que pedem menos Pizzas. Todos pedem sobremesa.
# MAGIC
# MAGIC - Cluster 2 - Clientes que pedem mais Pizzas. Todos pedem sobremesa em volume maior.
# MAGIC
# MAGIC Como estratégia de Marketing, poderíamos oferecer ao cliente uma sobremesa grátis no caso de comprar mais uma Pizza de maior valor. Com base na Segmentação provavelmente essa estratégia teria sucesso.
# MAGIC
# MAGIC Com base no exemplo acima, você conseguiria criar as seguintes Segmentações?
# MAGIC
# MAGIC - Segmentação 2 - Variáveis Pizza e Salada
# MAGIC - Segmentação 3 - Variáveis Pizza e Localidade
# MAGIC - Segmentação 4 - Variáveis Pizza, Salada e Localidade
# MAGIC - Segmentação 5 - Variáveis Pizza, Salada e Sobremesa

# COMMAND ----------

# MAGIC %md
# MAGIC #### Segmentação 2

# COMMAND ----------

# Usaremos duas variáveis
X1 = df[['pizza' , 'salada']].iloc[: , :].values

# Lista de valores de Inertia (Inertia e WCSS são a mesma coisa)
inertia = []

# Loop para testar os valores de K
for n in range(2 , 11):
    modelo = (KMeans(n_clusters = n,
                     init = 'k-means++', 
                     n_init = 10,
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan'))
    modelo.fit(X1)
    inertia.append(modelo.inertia_)

# Plot
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , 11) , inertia , 'o')
plt.plot(np.arange(2 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('Inertia')
plt.show()

# COMMAND ----------

# Criação do modelo com 3 clusters
modelo_seg2 = (KMeans(n_clusters = 3,
                      init = 'k-means++', 
                      n_init = 10 ,
                      max_iter = 300, 
                      tol = 0.0001,  
                      random_state = 111  , 
                      algorithm = 'elkan') )

# Treinamento do modelo
modelo_seg2.fit(X1)

# Labels
labels2 = modelo_seg2.labels_

# Centróides
centroids2 = modelo_seg2.cluster_centers_

# COMMAND ----------

# Plot

# Parâmetros do Meshgrid
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = modelo_seg2.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15, 7) )
plt.clf()
Z = Z.reshape(xx.shape)

# Plot da imagem
plt.imshow(Z, 
           interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Dark2, 
           aspect = 'auto', 
           origin = 'lower')

# Plot dos pontos de dados
plt.scatter( x = 'pizza', y = 'salada', data = df, c = labels2, s = 200 )
plt.scatter(x = centroids2[: , 0], y =  centroids2[: , 1], s = 300, c = 'red', alpha = 0.5)
plt.xlabel('Pizza')
plt.ylabel('Salada')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretação**: 
# MAGIC
# MAGIC - O ponto vermelho é o centróide de cada cluster (segmento).
# MAGIC
# MAGIC
# MAGIC - No cluster 1 (área em cinza) temos os clientes que pediram menos Pizzas e mais Saladas.
# MAGIC
# MAGIC
# MAGIC - No cluster 2 (área em verde escuro) temos os clientes que pediram poucas Pizzas e poucas Saladas.
# MAGIC
# MAGIC
# MAGIC - No cluster 3 (área em verde claro) estão clientes que pediram mais Pizzas e menos Saladas.
# MAGIC
# MAGIC **Análise**:
# MAGIC
# MAGIC Os clusters 1 e 3 são de clientes com comportamentos opostos. A equipe de Marketing poderia concentrar os esforços nos clientes do cluster 2, pois são clientes que compram Pizzas e Saladas e, portanto, tendem a consumir mais itens variados evitando manter os estoques cheios de um único item. 
# MAGIC
# MAGIC Ou então, concentrar os esforços nos clientes que consomem produtos que geram mais lucro. Teríamos que verificar qual item, Pizza ou Salada, é mais rentável.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segmentação 3
# MAGIC
# MAGIC Segmentação 3 - Variáveis Pizza e Localidade

# COMMAND ----------

# Filtrando o dataframe por colunas 
df_item_pedidos_idx[['index', 'bebida', 'pizza', 'sobremesa', 'salada', 'localidade']]

# COMMAND ----------

# Criando um novo dataframe
df2 = df_item_pedidos_idx[['index', 'bebida', 'pizza', 'sobremesa', 'salada', 'localidade']]

# COMMAND ----------

# Resumo do dataset
df2.describe()

# COMMAND ----------

# Usaremos duas variáveis
X1 = df2[['pizza' , 'localidade']].iloc[: , :].values

# Lista de valores de Inertia (Inertia e WCSS são a mesma coisa)
inertia = []

# Loop para testar os valores de K
for n in range(2 , 11):
    modelo = (KMeans(n_clusters = n,
                     init = 'k-means++', 
                     n_init = 10,
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan'))
    modelo.fit(X1)
    inertia.append(modelo.inertia_)

# Plot
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , 11) , inertia , 'o')
plt.plot(np.arange(2 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('Inertia')
plt.show()

# COMMAND ----------

# Criação do modelo com 4 clusters
modelo_seg3 = (KMeans(n_clusters = 4,
                      init = 'k-means++', 
                      n_init = 10 ,
                      max_iter = 300, 
                      tol = 0.0001,  
                      random_state = 111  , 
                      algorithm = 'elkan') )

# Treinamento do modelo
modelo_seg3.fit(X1)

# Labels
labels3 = modelo_seg3.labels_

# Centróides
centroids3 = modelo_seg3.cluster_centers_

# COMMAND ----------

# Plot

# Parâmetros do Meshgrid
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = modelo_seg3.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15, 7) )
plt.clf()
Z = Z.reshape(xx.shape)

# Plot da imagem
plt.imshow(Z, 
           interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel1, 
           aspect = 'auto', 
           origin = 'lower')

# Plot dos pontos de dados
plt.scatter( x = 'pizza', y = 'localidade', data = df2, c = labels3, s = 200 )
plt.scatter(x = centroids3[: , 0], y =  centroids3[: , 1], s = 300, c = 'red', alpha = 0.5)
plt.xlabel('Pizza')
plt.ylabel('Localidade')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretação**: 
# MAGIC
# MAGIC - O ponto vermelho é o centróide de cada cluster (segmento).
# MAGIC
# MAGIC
# MAGIC - Observe que os clusters da esquerda no gráfico contém os pedidos de todas as Localidades, mas com menor número de Pizzas. Já os clusters da direita no gráfico contém pedidos de todas as Localidades com com maior número de Pizzas.
# MAGIC
# MAGIC **Análise**:
# MAGIC
# MAGIC Queremos aumentar as vendas, certo? Então teríamos que investigar mais a fundo os pedidos dos clusters à esquerda do gráfico e compreender em mais detalhes as características desses pedidos e que tipo de oferta podemos fazer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segmentação 4
# MAGIC
# MAGIC Segmentação 4 - Variáveis Pizza, Salada e Localidade

# COMMAND ----------

# Usaremos três variáveis
X1 = df2[['pizza' , 'salada' , 'localidade']].iloc[: , :].values

# Lista de valores de Inertia (Inertia e WCSS são a mesma coisa)
inertia = []

# Loop para testar os valores de K
for n in range(2 , 11):
    modelo = (KMeans(n_clusters = n,
                     init = 'k-means++', 
                     n_init = 10,
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan'))
    modelo.fit(X1)
    inertia.append(modelo.inertia_)

# Plot
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , 11) , inertia , 'o')
plt.plot(np.arange(2 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('Inertia')
plt.show()

# COMMAND ----------

# Criação do modelo com 4 clusters
modelo_seg4 = (KMeans(n_clusters = 4,
                      init = 'k-means++', 
                      n_init = 10 ,
                      max_iter = 300, 
                      tol = 0.0001,  
                      random_state = 111  , 
                      algorithm = 'elkan') )

# Treinamento do modelo
modelo_seg4.fit(X1)

# Labels
labels4 = modelo_seg4.labels_

# Centróides
centroids4 = modelo_seg4.cluster_centers_

# COMMAND ----------

# Instala o Plotly
!pip install -q plotly

# COMMAND ----------

# Pacotes para o gráfico 3D
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected = True)

# COMMAND ----------

# Versões dos pacotes usados neste jupyter notebook
%reload_ext watermark
%watermark -a "Eng. Diego de Souza" --iversions

# COMMAND ----------

# Plot

# Gráfico 3D
grafico = go.Scatter3d(x = df2['pizza'], 
                       y = df2['salada'], 
                       z = df2['localidade'], 
                       mode = 'markers', 
                       marker = dict(color = labels4, 
                                     size = 4,
                                     line = dict(color = labels4, width = 15),
                                     opacity = 0.7))

# Layout do gráfico
layout = go.Layout(title = 'Clusters',
                   scene = dict(xaxis = dict(title  = 'Pizza'),
                                yaxis = dict(title  = 'Salada'),
                                zaxis = dict(title  = 'Localidade')))

# Plot da figura (gráfico + layout)
fig = go.Figure(data = grafico, layout = layout)
py.offline.iplot(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretação**: 
# MAGIC
# MAGIC - Observamos 2 clusters inferiores e 2 superiores.
# MAGIC
# MAGIC
# MAGIC - Cada ponto de dado representa uma coordenada de 3 dimensões.
# MAGIC
# MAGIC **Análise**:
# MAGIC
# MAGIC Aqui o ideal é avaliar o gráfico de forma interativa aproveitando essa propriedade do Plotly.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segmentação 5
# MAGIC
# MAGIC Segmentação 5 - Variáveis Pizza, Salada e Sobremesa

# COMMAND ----------

# Usaremos três variáveis
X1 = df2[['pizza' , 'salada' , 'sobremesa']].iloc[: , :].values

# Lista de valores de Inertia (Inertia e WCSS são a mesma coisa)
inertia = []

# Loop para testar os valores de K
for n in range(2 , 11):
    modelo = (KMeans(n_clusters = n,
                     init = 'k-means++', 
                     n_init = 10,
                     max_iter = 300, 
                     tol = 0.0001,  
                     random_state = 111, 
                     algorithm = 'elkan'))
    modelo.fit(X1)
    inertia.append(modelo.inertia_)

# Plot
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , 11) , inertia , 'o')
plt.plot(np.arange(2 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('Inertia')
plt.show()

# COMMAND ----------

# Criação do modelo com 2 clusters
modelo_seg5 = (KMeans(n_clusters = 2,
                      init = 'k-means++', 
                      n_init = 10 ,
                      max_iter = 300, 
                      tol = 0.0001,  
                      random_state = 111  , 
                      algorithm = 'elkan') )

# Treinamento do modelo
modelo_seg5.fit(X1)

# Labels
labels5 = modelo_seg5.labels_

# Centróides
centroids5 = modelo_seg5.cluster_centers_

# COMMAND ----------

# Plot

# Gráfico 3D
grafico = go.Scatter3d(x = df2['pizza'], 
                       y = df2['salada'], 
                       z = df2['sobremesa'], 
                       mode = 'markers', 
                       marker = dict(color = labels5, 
                                     size = 4,
                                     line = dict(color = labels5, width = 15),
                                     opacity = 0.7))

# Layout do gráfico
layout = go.Layout(title = 'Clusters',
                   scene = dict(xaxis = dict(title  = 'Pizza'),
                                yaxis = dict(title  = 'Salada'),
                                zaxis = dict(title  = 'Sobremesa')))

# Plot da figura (gráfico + layout)
fig = go.Figure(data = grafico, layout = layout)
py.offline.iplot(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretação**: 
# MAGIC
# MAGIC - Observamos a clara separação entre os dados dois 2 clusters.
# MAGIC
# MAGIC
# MAGIC - Cada ponto de dado representa uma coordenada de 3 dimensões.
# MAGIC **Análise**:
# MAGIC
# MAGIC Aqui o ideal é avaliar o gráfico de forma interativa aproveitando essa propriedade do Plotly.

# COMMAND ----------

# MAGIC %md
# MAGIC **Exemplo de Relatório final (Considerando a Segmentação 5)**

# COMMAND ----------

# Shape dos labels
labels5.shape

# COMMAND ----------

# Tipo
type(labels5)

# COMMAND ----------

# Converte o array para dataframe
df_labels = pd.DataFrame(labels5)

# COMMAND ----------

# Visualiza
df_labels.head(5)

# COMMAND ----------

# Tipo
type(df_labels)

# COMMAND ----------

# Vamos fazer o merge de df2 e os labels (clusters) encontrados pelo modelo
# Lembre-se que usamos somente 3 variáveis para criar a segmentação
df_final = df2.merge(df_labels, left_index = True, right_index = True)

# COMMAND ----------

# Visualiza
df_final

# COMMAND ----------

# Ajusta o nome da coluna
df_final.rename(columns = {0:"cluster"}, inplace = True)

# COMMAND ----------

# Visualiza
df_final

# COMMAND ----------

# MAGIC %md
# MAGIC ### End!
